"""K-shot sampling + dataset registration for CD-FSOD targets (COCO-format)."""
import json, random
from collections import defaultdict
from detectron2.data.datasets import register_coco_instances


def make_kshot(ann_file, out_file, k, seed):
    """Pick K annotated instances per class; keep only those annotations.
    Same (k, seed) -> same support set, so all experiments are comparable."""
    coco = json.load(open(ann_file))
    rng = random.Random(seed)

    by_cls = defaultdict(list)
    for a in coco["annotations"]:
        by_cls[a["category_id"]].append(a)

    chosen = []
    for cid in sorted(by_cls):
        anns = by_cls[cid][:]
        rng.shuffle(anns)
        chosen += anns[:k]

    img_ids = {a["image_id"] for a in chosen}
    out = {
        "images": [i for i in coco["images"] if i["id"] in img_ids],
        "annotations": chosen,
        "categories": coco["categories"],
    }
    json.dump(out, open(out_file, "w"))
    print(f"[k-shot] k={k} seed={seed}: {len(chosen)} anns, {len(img_ids)} images -> {out_file}")
    return out_file


def register(name, img_dir, ann_file):
    register_coco_instances(name, {}, ann_file, img_dir)
