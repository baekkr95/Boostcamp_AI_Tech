from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import (build_dataloader, build_dataset, replace_ImageToTensor)

import pprint
import json
import wandb
import math
import datetime

now = datetime.datetime.now()
nowDatetime = now.strftime('%Y-%m-%d_%H:%M')


wandb.login(key='38e2f5a0d3af46ebb74291222733447068d4d9b9')

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

num_data = dict(
    train=4883, #test=,
    cv_train_1=3901, cv_val_1=982,
    cv_train_2=3902, cv_val_2=981,
    cv_train_3=3878, cv_val_3=1005,
    cv_train_4=3890, cv_val_4=993,
    cv_train_5=3961, cv_val_5=922,
)


# config 파일 찾음
'''
config 폴더 안에 있는 .py 파일에 우리의 _base_ 코드들을 찾는 리스트가 있음.
[
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
'''


# setup
model_name = 'faster_rcnn_r50_fpn_1x_coco'
job_name = nowDatetime


cfg = Config.fromfile(f'./config/{model_name}.py')


# directory
cfg.data_root = '../../../../dataset/'
cfg.json_root = '../../../../stratified_kfold/'
cfg.work_dir = './work_dirs/baseline_dir/'

cfg.fold = 1
cfg.seed = 42
cfg.gpu_ids = [0]
cfg.img_size = 512

cfg.runner.max_epochs = 1  # epochs
cfg.data.workers_per_gpu = 4  # num_worker
cfg.data.samples_per_gpu = 16  # batch_size


# dataset
cfg.data.train.classes = classes    # classes 지정 필수!!
cfg.data.train.img_prefix = cfg.data_root
cfg.data.train.ann_file = cfg.json_root + f'cv_train_{cfg.fold}.json'
cfg.train_pipeline[2]['img_scale'] = (cfg.img_size, cfg.img_size)
cfg.data.train.pipeline = cfg.train_pipeline

cfg.data.val.classes = classes
cfg.data.val.img_prefix = cfg.data_root
cfg.data.val.ann_file = cfg.json_root + f'cv_val_{cfg.fold}.json'
cfg.data.val.pipeline[1]['img_scale'] = (cfg.img_size, cfg.img_size)


cfg.evaluation = dict(
    interval=1,
    metric='bbox',
    classwise=True,
    iou_thrs=[0.50],
    metric_items=['mAP','mAP_s','mAP_m','mAP_l']
)


# model
cfg.model.roi_head.bbox_head.num_classes = len(classes)

cfg.optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.01)
cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=math.ceil(num_data[f'cv_train_{cfg.fold}'] / (cfg.data.samples_per_gpu)),
    warmup_ratio=0.001,
    min_lr=1e-07,
)

# log (wandb)
cfg.log_config.interval = math.ceil(num_data[f'cv_train_{cfg.fold}'] / (cfg.data.samples_per_gpu*10))
cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='WandbLoggerHook',
         init_kwargs=dict(
            project='mmdetection',
            name=f'exp-{model_name}-job-{job_name}',
            entity='baekkr95'
        )
    )
]


# plot cfg
for i, k in enumerate(cfg.keys()):
    print(i, k)
    pprint.pprint(cfg[k])


# start
set_random_seed(cfg.seed, deterministic=True)

datasets = []
datasets.append(build_dataset(cfg.data.train))

model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'))
model.init_weights()

train_detector(model, datasets[0], cfg, distributed=False, validate=True)