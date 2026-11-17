"""Prototype-based few-shot detector baseline (no training).

  proposals : class-agnostic RPN of the COCO-pretrained Faster R-CNN (Detectron2)
  features  : frozen DINOv2 patch features, RoIAlign-pooled per region
  classifier: cosine similarity to class prototypes built from the K-shot support set
  score     : RPN objectness^alpha * softmax(cos_sim / tau)

Usage (Colab):
  python proto_baseline.py --root /content/drive/MyDrive/clipart1k --k 5
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.ops import roi_align, batched_nms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(3, 1, 1)
PATCH = 14  # DINOv2 patch size


# ---------------------------------------------------------------- proposals
def load_rpn(topk):
    name = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(name))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(name)
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = topk
    cfg.MODEL.DEVICE = DEV
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    return model.eval()


@torch.no_grad()
def get_proposals(rpn, img):
    """Returns boxes (xyxy, original image coords) and objectness probabilities."""
    w, h = img.size
    s = 800 / min(h, w)
    if max(h, w) * s > 1333:
        s = 1333 / max(h, w)
    nw, nh = int(round(w * s)), int(round(h * s))
    x = np.asarray(img.resize((nw, nh), Image.BILINEAR))[:, :, ::-1].copy()  # BGR
    x = torch.from_numpy(x).permute(2, 0, 1).float().to(DEV)
    images = rpn.preprocess_image([{"image": x}])
    feats = rpn.backbone(images.tensor)
    props, _ = rpn.proposal_generator(images, feats, None)
    p = props[0]
    scale = torch.tensor([nw / w, nh / h, nw / w, nh / h], device=DEV)
    return p.proposal_boxes.tensor / scale, p.objectness_logits.sigmoid()


# ----------------------------------------------------------------- features
@torch.no_grad()
def patch_features(dino, img, short=448, max_side=896):
    w, h = img.size
    s = short / min(h, w)
    if max(h, w) * s > max_side:
        s = max_side / max(h, w)
    nw = max(PATCH, int(round(w * s / PATCH)) * PATCH)
    nh = max(PATCH, int(round(h * s / PATCH)) * PATCH)
    x = np.asarray(img.resize((nw, nh), Image.BICUBIC))
    x = torch.from_numpy(x.copy()).permute(2, 0, 1).float().to(DEV) / 255
    x = ((x - MEAN) / STD)[None]
    tok = dino.forward_features(x)["x_norm_patchtokens"]            # [1, N, D]
    feat = tok.reshape(1, nh // PATCH, nw // PATCH, -1).permute(0, 3, 1, 2)
    return feat, (nw / w, nh / h)                                   # [1, D, h', w']


def region_embed(feat, scale, boxes):
    """RoIAlign-pool patch features inside each box -> L2-normalised embedding."""
    sx, sy = scale
    b = boxes * torch.tensor([sx, sy, sx, sy], device=boxes.device)
    pooled = roi_align(feat, [b], output_size=(3, 3), spatial_scale=1.0 / PATCH,
                       sampling_ratio=2, aligned=True)               # [R, D, 3, 3]
    return F.normalize(pooled.mean((2, 3)), dim=-1)                  # [R, D]


# --------------------------------------------------------------- prototypes
def build_prototypes(dino, root, shot_json):
    d = json.load(open(shot_json))
    cats = sorted(c["id"] for c in d["categories"])
    imgs = {i["id"]: i for i in d["images"]}
    embs = {c: [] for c in cats}
    for a in d["annotations"]:
        img = Image.open(f"{root}/train/{imgs[a['image_id']]['file_name']}").convert("RGB")
        feat, sc = patch_features(dino, img)
        x, y, w, h = a["bbox"]
        box = torch.tensor([[x, y, x + w, y + h]], dtype=torch.float32, device=DEV)
        embs[a["category_id"]].append(region_embed(feat, sc, box)[0])
    assert all(len(v) > 0 for v in embs.values()), "a class has no support instance"
    protos = torch.stack([F.normalize(torch.stack(embs[c]).mean(0), dim=-1) for c in cats])
    return protos, cats                                              # [C, D]


# --------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--backbone", default="dinov2_vitb14",
                   help="dinov2_vits14 | dinov2_vitb14 | dinov2_vitl14")
    p.add_argument("--proposals", type=int, default=300)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--nms", type=float, default=0.5)
    p.add_argument("--out", default="runs")
    a = p.parse_args()

    rpn = load_rpn(a.proposals)
    dino = torch.hub.load("facebookresearch/dinov2", a.backbone, trust_repo=True).to(DEV).eval()

    protos, cats = build_prototypes(dino, a.root, f"{a.root}/annotations/{a.k}_shot.json")
    print(f"prototypes: {tuple(protos.shape)}")

    gt = COCO(f"{a.root}/annotations/test.json")
    results = []
    img_ids = sorted(gt.getImgIds())
    for n, iid in enumerate(img_ids):
        info = gt.loadImgs(iid)[0]
        img = Image.open(f"{a.root}/test/{info['file_name']}").convert("RGB")
        boxes, obj = get_proposals(rpn, img)
        if len(boxes) == 0:
            continue
        feat, sc = patch_features(dino, img)
        emb = region_embed(feat, sc, boxes)                          # [P, D]
        probs = ((emb @ protos.T) / a.tau).softmax(-1)               # [P, C]
        scores = (obj[:, None] ** a.alpha) * probs                   # [P, C]
        top, idx = scores.flatten().topk(min(1000, scores.numel()))
        pi, ci = idx // len(cats), idx % len(cats)
        keep = batched_nms(boxes[pi], top, ci, a.nms)[:100]
        for j in keep.tolist():
            x1, y1, x2, y2 = boxes[pi[j]].tolist()
            results.append({"image_id": iid, "category_id": cats[ci[j]],
                            "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(top[j])})
        if (n + 1) % 50 == 0:
            print(f"{n + 1}/{len(img_ids)} images")

    ev = COCOeval(gt, gt.loadRes(results), "bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    m = {"AP": ev.stats[0] * 100, "AP50": ev.stats[1] * 100, "AP75": ev.stats[2] * 100,
         "args": vars(a)}
    out = f"{a.out}/proto_{a.backbone}_{a.k}shot"
    os.makedirs(out, exist_ok=True)
    json.dump(m, open(f"{out}/metrics.json", "w"), indent=2)
    print(f"\nPROTOTYPE BASELINE  k={a.k}  AP={m['AP']:.2f}  AP50={m['AP50']:.2f}  AP75={m['AP75']:.2f}")


if __name__ == "__main__":
    main()
