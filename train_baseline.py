"""Fine-tune COCO-pretrained Faster R-CNN on K-shot target data, evaluate on test split.

Usage (Colab):
  python train_baseline.py --root /content/clipart1k --k 5 --seed 0
Expected layout:  <root>/train/ + <root>/annotations/train.json
                  <root>/test/  + <root>/annotations/test.json
"""
import argparse, json, os
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from data import make_kshot, register


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, name):
        return COCOEvaluator(name, output_dir=cfg.OUTPUT_DIR)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--out", default="runs")
    a = p.parse_args()

    ann = f"{a.root}/annotations"
    shot_json = make_kshot(f"{ann}/train.json", f"{ann}/train_{a.k}shot_s{a.seed}.json", a.k, a.seed)
    tr, te = f"tgt_train_{a.k}_{a.seed}", "tgt_test"
    register(tr, f"{a.root}/train", shot_json)
    register(te, f"{a.root}/test", f"{ann}/test.json")

    n_cls = len(json.load(open(shot_json))["categories"])
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # source domain = COCO
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = n_cls
    cfg.MODEL.BACKBONE.FREEZE_AT = 5          # freeze backbone (few-shot -> avoid overfitting)
    cfg.DATASETS.TRAIN, cfg.DATASETS.TEST = (tr,), (te,)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = a.iters
    cfg.SOLVER.STEPS = (int(a.iters * 0.8),)
    cfg.SOLVER.WARMUP_ITERS = 50
    cfg.SOLVER.CHECKPOINT_PERIOD = 10**9
    cfg.OUTPUT_DIR = f"{a.out}/frcnn_{a.k}shot_s{a.seed}"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    ev = COCOEvaluator(te, output_dir=cfg.OUTPUT_DIR)
    res = inference_on_dataset(trainer.model, build_detection_test_loader(cfg, te), ev)
    print(res["bbox"])
    json.dump(res["bbox"], open(f"{cfg.OUTPUT_DIR}/metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
