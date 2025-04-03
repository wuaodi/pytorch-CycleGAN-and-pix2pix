Implement of ***"Cross-domain Spacecraft Component Segmentation Based on Edge Consistency Generative Neural Network"***.

**Authors**: Aodi Wu, Jianhong Zuo, Shengyang Zhang and Xue Wan

Accepted by The 17th International Conference onDigital Image Processing (**ICDIP 2025**)

# core code
```
# edge loss
models/edgeloss.py
models/cycle_gan_model.py

# yolo bbox results to SAM
SAM/bbox_to_seg.py
```

# training and testing

refering to:

CycleGAN: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

YOLO: https://github.com/ultralytics/yolov5

SAM: https://github.com/facebookresearch/segment-anything
