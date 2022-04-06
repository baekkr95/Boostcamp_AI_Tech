# mmdet_train.py
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import (build_dataloader, build_dataset, replace_ImageToTensor)

import os
import json
import math

import pytz
import datetime

import gc
import torch

import re
import glob
from pathlib import Path


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.
    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [
            re.search(rf"%s(\d+)" % path.stem, d) for d in dirs
        ]  
        i = [int(m.groups()[0]) for m in matches if m]  
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def make_cfg(config_file, work_dir, fold):

    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
               "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
    
    # fold 별 이미지 mean, std
    data_info = dict(
        train=dict(num=4883, mean=[110.0761, 117.3985, 123.6527], std=[54.7579, 53.3471, 53.9986]),
        test=dict(num=4871, mean=[109.7817, 117.1358, 123.2762], std=[55.0127, 53.6474, 54.1965]),
        cv_train_1=dict(num=3901, mean=[110.0174, 117.2752, 123.485], std=[54.664 , 53.2387, 53.9031]),
        cv_val_1=dict(num=982),
        cv_train_2=dict(num=3902, mean=[110.115 , 117.4417, 123.6848], std=[54.8147, 53.4252, 54.0498]),
        cv_val_2=dict(num=981),
        cv_train_3=dict(num=3878, mean=[109.8931, 117.3784, 123.6862], std=[54.8974, 53.4674, 54.1553]),
        cv_val_3=dict(num=1005),
        cv_train_4=dict(num=3890, mean=[109.9669, 117.1712, 123.4992], std=[54.6633, 53.2836, 53.9122]),
        cv_val_4=dict(num=993),
        cv_train_5=dict(num=3961, mean=[110.3819, 117.7205, 123.904], std=[54.7507, 53.3212, 53.9737]),
        cv_val_5=dict(num=922),
    )
    base_root = '/opt/ml/detection/'

    # setup
    cfg = Config.fromfile(base_root + f'baseline/mmdetection/configs/_boost_/config/{config_file}.py')

    cfg.data_root = base_root + 'dataset/'
    cfg.work_dir = work_dir + f'/fold{fold}'
    os.makedirs(cfg.work_dir, exist_ok=True)

    cfg.fold = fold
    cfg.seed = 42
    cfg.gpu_ids = [0]
    cfg.runner.max_epochs = 30
    cfg.data.samples_per_gpu = 8
    cfg.data.workers_per_gpu = 4
    cfg.fp16 = dict(loss_scale=512.)

    # main에서 fold에 조건 걸었음. 0이면 전체 데이터셋, 3이면 fold 3으로 cv 진행
    if cfg.fold == 0:
        cfg.json_root = base_root + 'dataset/'
        train_json = 'train'
        val_json = None
        cfg.img_size = 1024
    else:
        cfg.json_root = base_root + 'stratified_kfold/'
        train_json = f'cv_train_{cfg.fold}'
        val_json = f'cv_val_{cfg.fold}'
        cfg.img_size = 512


    albu_train_transforms = [
    dict(
        type='OneOf',
        transforms=[
            dict(type='Flip',p=1.0),
            dict(type='RandomRotate90',p=1.0)
        ], p=0.5),
    dict(
        type = 'OneOf', # - channel dropout, channel shuffle
        transforms = [
            dict(type='ChannelDropout',p=0.5),
            dict(type='ChannelShuffle', p=0.5)
        ],
    p=0.1),
    dict(type='CLAHE',p=0.5),
    dict(type='RandomShadow',p=0.1),
    dict(type='RandomResizedCrop',height=512, width=512, scale=(0.5, 1.0), p=0.5),
    dict(type='RandomBrightnessContrast',brightness_limit=0.1, contrast_limit=0.15, p=0.5),
    dict(type='HueSaturationValue', hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=10, p=0.5),
    dict(type='GaussNoise', p=0.3),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', p=1.0),
            dict(type='GaussianBlur', p=1.0),
            dict(type='MedianBlur', blur_limit=5, p=1.0),
            dict(type='MotionBlur', p=1.0)
        ], p=0.1)
    ]


    # # augmentation
    # cfg.train_pipeline = [
    #     {'type': 'LoadImageFromFile'},
    #     {'type': 'LoadAnnotations', 'with_bbox': True},
    #     {'type': 'Resize', 'img_scale': (cfg.img_size, cfg.img_size), 'keep_ratio': True},
    #     {'type': 'RandomFlip', 'flip_ratio': 0.5},
    #     {'type': 'CutOut', 'n_holes': (10,20), 'cutout_shape': [(4,4), (4,8), (8,4)]},

    #     {'type':'Albu',
    #     'transforms':albu_train_transforms,
    #     'bbox_params':{
    #         'type':'BboxParams',
    #         'format':'pascal_voc',
    #         'label_fields':['gt_labels'],
    #         'min_visibility':0.0,
    #         'filter_lost_elements':True},
    #     'keymap':{
    #         'img': 'image',
    #         # 'gt_masks': 'masks',
    #         'gt_bboxes': 'bboxes'
    #         },
    #     'update_pad_shape':False,
    #     'skip_img_without_anno':False},

    #     {'type': 'Normalize',
    #      'mean': data_info[train_json]['mean'],
    #      'std': data_info[train_json]['std'],
    #      'to_rgb': True},
    #     {'type': 'Pad', 'size_divisor': 32},
    #     {'type': 'DefaultFormatBundle'},
    #     {'type': 'Collect', 'keys': ['img', 'gt_bboxes', 'gt_labels']}
    # ]
    cfg.val_pipeline = [
        {'type': 'LoadImageFromFile'},
        {'type': 'MultiScaleFlipAug',
         'img_scale': (cfg.img_size, cfg.img_size),
         'flip': False,
         'transforms': [{'type': 'Resize', 'keep_ratio': True},
                        {'type': 'RandomFlip'},
                        {'type': 'Normalize',
                         'mean': data_info[train_json]['mean'],
                         'std': data_info[train_json]['std'],
                         'to_rgb': True},
                        {'type': 'Pad', 'size_divisor': 32},
                        {'type': 'ImageToTensor', 'keys': ['img']},
                        {'type': 'Collect', 'keys': ['img']}]}
    ]
    
    # base_mean = [123.675, 116.28, 103.53]
    # base_std = [58.395, 57.12, 57.375]
    # cfg.train_pipeline[5]['mean'] = base_mean
    # cfg.train_pipeline[5]['std'] = base_std
    # cfg.val_pipeline[1]['transforms'][2]['mean'] = base_mean
    # cfg.val_pipeline[1]['transforms'][2]['std'] = base_std

    # dataset
    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = cfg.data_root
    cfg.data.train.ann_file = cfg.json_root + train_json + '.json'
    # cfg.data.train.pipeline = cfg.train_pipeline

    if cfg.fold != 0:
        cfg.data.val.classes = classes
        cfg.data.val.img_prefix = cfg.data_root
        cfg.data.val.ann_file = cfg.json_root + val_json + '.json'
        cfg.data.val.pipeline = cfg.val_pipeline   # val_pipeline

    # config
    # cfg.optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
    # cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    # cfg.checkpoint_config = dict(interval=1, max_keep_ckpts=1)
    cfg.evaluation = dict(
        interval=1,
        metric='bbox',
        classwise=True,
        iou_thrs=[0.50],
        metric_items=['mAP','mAP_s','mAP_m','mAP_l']
    )

    one_epoch = math.ceil(data_info[train_json]['num'] / cfg.data.samples_per_gpu)
    # cfg.lr_config = dict(
    #     policy='step',  # CosineAnnealing
    #     warmup='linear',
    #     warmup_iters=one_epoch,
    #     warmup_ratio=0.001,
    #     # min_lr=1e-6,  # CosineAnnealing
    #     step=[5,8]
    # )

    exp = work_dir.split('/')[-1]
    # cfg.log_config.interval = one_epoch
    cfg.log_config.hooks = [
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(project='object_detection_baekkr',
                              name=f'{exp}-fold{cfg.fold}-{config_file}',
                              entity='mg_generation',
                              tags=[config_file],
                              group=exp,
                              job_type=f'fold{cfg.fold}')
        )
    ]

    return cfg


if __name__ == '__main__':
    for i in range(1):
        for fold in [3]:

            gc.collect()
            torch.cuda.empty_cache()

            work_dir = increment_path('/opt/ml/detection/work_dirs/mmdet_exp')

            now = datetime.datetime.now(pytz.timezone('Asia/Seoul'))
            timestamp = now.strftime('%Y%m%d_%H%M%S')


            '''
            config_file 목록
            - faster_rcnn_r50_fpn_1x_coco
            - cascade_rcnn_r50_fpn_1x_coco
            - swin_rcnn_fpn_1x_coco
            - retinanet_r50_fpn_1x_coco
            - detr_r50_8x2_150e_coco
            - cornernet_hourglass104_mstest_8x6_210e_coco
            - centernet_resnet18_dcnv2_140e_coco
            '''

            cfg = make_cfg(config_file='faster_rcnn_r50_fpn_1x_coco', work_dir=work_dir, fold=fold)
            #cfg.runner.max_epochs = 20
            cfg.model.train_cfg.rpn_proposal.nms = dict(type='nms', iou_threshold=0.5)

            # for문 i 별로 비교
            # 모델별로 cfg가 상이하니까 주의할 것
            # if i==0:
            #     cfg.model.train_cfg.rpn_proposal.nms = dict(type='nms', iou_threshold=0.6)
            # elif i==1:
            #     cfg.model.train_cfg.rpn_proposal.nms = dict(type='nms', iou_threshold=0.5)
            


            cfg.dump(cfg.work_dir+'/train_config.py')
            #cfg.log_config['hooks'][1]['init_kwargs']['config'] = cfg

            set_random_seed(cfg.seed, deterministic=True)

            datasets = []
            datasets.append(build_dataset(cfg.data.train))

            model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'))
            model.init_weights()

            meta = dict()
            meta['config'] = cfg.pretty_text

            # cv valid 조건
            if fold == 0:
                validate = False
            else:
                validate = True
            
            train_detector(
                model, datasets[0], cfg, distributed=False,
                validate=validate, timestamp=timestamp, meta=meta
            )