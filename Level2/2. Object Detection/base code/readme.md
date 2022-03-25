## MMDetection Baseline Code
### 베이스라인 파일 실행
- `python baseline.py` 명령어로 터미널에서 실행
- baseline.py이 config폴더에서 실행할 파일을 찾고, 
- config에서 \_base_ 폴더의 datasets, models, schedules, default_runtime.py을 참조해서 학습을 시작함

```bash
mmdetection basecode
│ └─baseline.py
├─_base_
│  ├─datasets
│  │  ├─coco_detection.py
│  │  └─coco_detection_album.py
│  ├─models
│  │  ├─ccascade_rcnn_r50_fpn.py
│  │  └─htc_soft-faster_rcnn_r50_fpn.py
│  ├─schedules
│  │  └─schedule_1x.py
│  └─default_runtime.py
├─config
│  ├─cascade_rcnn_r50_fpn_1x_coco.py
│  ├─faster_rcnn_r50_fpn_1x_coco.py
│  └─swin_rcnn_fpn_1x_coco.py
├─wandb
└─baseline.py
```
