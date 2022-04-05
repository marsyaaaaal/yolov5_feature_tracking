# Detection-tracking Algorithm
This takes a yolo detection (https://github.com/ultralytics/yolov5) output and tracks it by using centroid tracking and pHash algortihm (http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html)

## Requirements:

Install Pytorch >= 1.7 here: https://pytorch.org/get-started/locally/

If gpu supported, install CUDA here: https://developer.nvidia.com/cuda-toolkit


## Guide:
Clone this repo and install requirements.txt

```bash
git clone https://github.com/marsyaaaaal/yolov5_feature_tracking.git

cd yolov5_feature_tracking

pip install requirements.txt
```

## Infrerence:

```bash
python detect.py --source data/vid.mp4 --weights yolov5s.pt --img 640 --agnostic-nms --conf-thres 0.25 --view-img
```