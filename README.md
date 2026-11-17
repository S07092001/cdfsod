# CD-FSOD starter (Phase 1-2: data + Faster R-CNN baseline)

## Colab setup (Runtime -> T4 GPU)
```
!pip install -q 'git+https://github.com/facebookresearch/detectron2.git' pycocotools
!git clone <your-github-repo> && cd <repo>
```

## Data
Get the target datasets from the CD-FSOD benchmark repo (lovelyqian/CDFSOD-benchmark on GitHub;
check its README for download links). Arrange each dataset as:
```
clipart1k/
  train/  test/
  annotations/train.json  annotations/test.json   # COCO format
```
Start with Clipart1k or NEU-DET (small).

## Run
```
python train_baseline.py --root /content/clipart1k --k 5 --seed 0
```
Repeat for k in {1,5,10} and seeds {0,1,2}; report mean +- std of AP / AP50 from metrics.json.

## Next
Phase 3: DE-ViT baseline, then add explanation-guided prototype refinement.
