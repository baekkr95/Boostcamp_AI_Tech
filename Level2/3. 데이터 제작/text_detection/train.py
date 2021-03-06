import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST

import gc
import yaml
import wandb
from functools import reduce, partial

import cv2
import json
from glob import glob

from sweep import update_args, get_sweep_cfg
from utils import increment_path, set_seeds, read_json
from custom_scheduler import CosineAnnealingWarmUpRestarts

from detect import detect
from deteval import calc_deteval_metrics
from inference import do_inference


def parse_args():
    parser = ArgumentParser()
    # directory
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/wj'))
    parser.add_argument('--val_data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/ICDAR17_valid_cv'))
    parser.add_argument('--json_dir', type=str,
                        default='/opt/ml/input/data/ICDAR17_train_cv/ufo/train.json', help='train json directory')
    parser.add_argument('--val_json_dir', type=str,
                        default='/opt/ml/input/data/ICDAR17_valid_cv/ufo/valid.json', help='valid json directory')
    parser.add_argument('--work_dir', type=str, default='./work_dirs',
                        help='the root dir to save logs and models about each experiment')
    # run environment
    parser.add_argument('--device', type=str, default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num_workers', type=int, default=8, help='dataloader num_workers')
    parser.add_argument('--save_interval', type=int, default=10, help='model save interval')
    parser.add_argument('--save_max_num', type=int, default=5, help='the max number of model save files')
    parser.add_argument('--train_eval', type=bool, default=False, help='boolean about evaluation on train dataset')     # ????????? false
    parser.add_argument('--eval_interval', type=int, default=1, help='evaluation metric log interval')                  # ??? epoch ?????? evaluation ??? ??????
    # training parameter
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--optm', type=str, default='adam')
    parser.add_argument('--schd', type=str, default='multisteplr')
    # etc
    parser.add_argument('--sweep', type=bool, default=False, help='sweep option')    # ????????? False??? ?????? ???

    args = parser.parse_args()
    if args.input_size % 32 != 0: raise ValueError('`input_size` must be a multiple of 32')
    return args


def do_training(
    data_dir, val_data_dir, json_dir, val_json_dir, work_dir, work_dir_exp,
    device, seed, num_workers, save_interval, save_max_num, train_eval, eval_interval,
    image_size, input_size, batch_size, learning_rate, max_epoch, optm, schd,
    sweep
    ):
    set_seeds(seed)
    
    # json_name can controlled by using args.json_dir
    json_name = json_dir.split('/')[-1].split('.')[0]
    dataset = SceneTextDataset(data_dir, split=json_name,
                               image_size=image_size, crop_size=input_size)
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    # val_json_name can controlled by using args.val_json_dir
    val_json_name = val_json_dir.split('/')[-1].split('.')[0]
    val_dataset = SceneTextDataset(val_data_dir, split=val_json_name,
                                   image_size=image_size, crop_size=input_size)
    val_dataset = EASTDataset(val_dataset)
    val_num_batches = math.ceil(len(val_dataset) / batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)

    # optimizer
    if optm == 'adam':
        if schd == 'cosignlr':
            optimizer = torch.optim.Adam(model.parameters(), lr=0)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optm == 'sgd':
        if schd == 'cosignlr':
            optimizer = torch.optim.SGD(model.parameters(), lr=0)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    # scheduler
    if schd == 'multisteplr':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)
    elif schd == 'reducelr':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
    elif schd == 'cosignlr':
        # when being used with CosineAnnealingWarmUpRestarts, optimizer must start from lr=0
        scheduler = CosineAnnealingWarmUpRestarts(
            optimizer, T_0=50, T_mult=1, eta_max=learning_rate, T_up=5, gamma=0.5)
    
    for epoch in range(max_epoch):
        # train
        model.train()
        epoch_loss, epoch_cls_loss, epoch_ang_loss, epoch_iou_loss = 0, 0, 0, 0
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {} Train]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_value = loss.item()
                epoch_loss += loss_value
                epoch_cls_loss += extra_info['cls_loss']
                epoch_ang_loss += extra_info['angle_loss']
                epoch_iou_loss += extra_info['iou_loss']

                pbar.update(1)
            pbar.set_postfix({
                'loss': epoch_loss/num_batches, 'cls_loss': epoch_cls_loss/num_batches,
                'ang_loss': epoch_ang_loss/num_batches, 'iou_loss': epoch_iou_loss/num_batches,
            })

        # train evaluation
        if train_eval is not False and (epoch + 1) % eval_interval == 0:
            gt_ufo = read_json(json_dir)
            # ckpt_fpath : already we have model, so we don't use ckpt
            # split : image folder name
            pred_ufo = do_inference(model=model, input_size=input_size, batch_size=batch_size,
                                    data_dir=data_dir, ckpt_fpath=None, split='images')
            
            epoch_precison, epoch_recall, epoch_hmean = do_evaluating(gt_ufo, pred_ufo)
            wandb.log({
                "train_metric/precision": epoch_precison/num_batches,
                "train_metric/recall": epoch_recall/num_batches,
                "train_metric/hmean": epoch_hmean/num_batches,
            }, commit=False)

        # valid
        model.eval()
        val_epoch_loss, val_epoch_cls_loss, val_epoch_ang_loss, val_epoch_iou_loss = 0, 0, 0, 0
        with tqdm(total=val_num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in val_loader:
                pbar.set_description('[Epoch {} Valid]'.format(epoch + 1))

                with torch.no_grad():
                    loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)

                loss_value = loss.item()
                val_epoch_loss += loss_value
                val_epoch_cls_loss += extra_info['cls_loss']
                val_epoch_ang_loss += extra_info['angle_loss']
                val_epoch_iou_loss += extra_info['iou_loss']

                pbar.update(1)
            pbar.set_postfix({
                'loss': val_epoch_loss/val_num_batches, 'cls_loss': val_epoch_cls_loss/val_num_batches,
                'ang_loss': val_epoch_ang_loss/val_num_batches, 'iou_loss': val_epoch_iou_loss/val_num_batches,
            })

        # valid evaluation
        if (epoch + 1) % eval_interval == 0:
            gt_ufo = read_json(val_json_dir)
            # ckpt_fpath : already we have model, so we don't use ckpt
            # split : image folder name
            pred_ufo = do_inference(model=model, input_size=input_size, batch_size=batch_size,
                                    data_dir=val_data_dir, ckpt_fpath=None, split='images')
            val_epoch_precison, val_epoch_recall, val_epoch_hmean = do_evaluating(gt_ufo, pred_ufo)
            wandb.log({
                "valid_metric/precision": val_epoch_precison/val_num_batches,
                "valid_metric/recall": val_epoch_recall/val_num_batches,
                "valid_metric/hmean": val_epoch_hmean/val_num_batches,
            }, commit=False)

        # ReduceLROnPlateau scheduler consider valid loss when doing step
        if schd == 'reducelr':
            scheduler.step(val_epoch_loss)
        else:
            scheduler.step()
        
        if (epoch + 1) % save_interval == 0:
            ckpt_fpath = osp.join(work_dir_exp, f'epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), ckpt_fpath)
            if len(glob(osp.join(work_dir_exp,'*.pth'))) > save_max_num:
                pth_list = [f.split('/')[-1].split('.')[0].split('_')[-1] for f in glob(osp.join(work_dir_exp,'*.pth'))]
                os.remove(osp.join(work_dir_exp, f'epoch_{sorted(pth_list, key=lambda x: int(x))[0]}.pth'))

        wandb.log({
            "train/loss": epoch_loss/num_batches, "valid/loss": val_epoch_loss/val_num_batches,
            "train/cls_loss": epoch_cls_loss/num_batches, "valid/cls_loss": val_epoch_cls_loss/val_num_batches,
            "train/ang_loss": epoch_ang_loss/num_batches, "valid/ang_loss": val_epoch_ang_loss/val_num_batches,
            "train/iou_loss": epoch_iou_loss/num_batches, "valid/iou_loss": val_epoch_iou_loss/val_num_batches,
        }, commit=True)  # commit=True : It notify that one epoch is ended with this log.  # default=True


def do_evaluating(gt_ufo, pred_ufo):
    epoch_precison, epoch_recall, epoch_hmean = 0, 0, 0
    num_images = len(gt_ufo['images'])
    # it calculate for each image
    for pred_image, gt_image in zip(sorted(pred_ufo['images'].items()), sorted(gt_ufo['images'].items())):
        pred_bboxes_dict, gt_bboxes_dict, gt_trans_dict = {}, {}, {}
        pred_bboxes_list, gt_bboxes_list, gt_trans_list = [], [], []

        # format change
        for pred_point in range(len(pred_image[1]['words'])):
            pred_bboxes_list.extend([pred_image[1]['words'][pred_point]['points']])
        pred_bboxes_dict[pred_image[0]] = pred_bboxes_list
        
        # format change
        for gt_point in range(len(gt_image[1]['words'])):
            gt_bboxes_list.extend([gt_image[1]['words'][str(gt_point)]['points']])
            gt_trans_list.extend([gt_image[1]['words'][str(gt_point)]['transcription']])
        gt_bboxes_dict[gt_image[0]] = gt_bboxes_list
        gt_trans_dict[gt_image[0]] = gt_trans_list
        
        # eval_metric['total'] : this consider all of bboxes in each image
        eval_metric = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict, transcriptions_dict=gt_trans_dict)
        epoch_precison += eval_metric['total']['precision']
        epoch_recall += eval_metric['total']['recall']
        epoch_hmean += eval_metric['total']['hmean']
    
    return epoch_precison, epoch_recall, epoch_hmean


def main(args):
    # generate work directory every experiment
    args.work_dir_exp = increment_path(osp.join(args.work_dir, 'exp'))
    if not osp.exists(args.work_dir_exp): os.makedirs(args.work_dir_exp)
    
    if args.sweep:
        # if you want to use tags, put tags=['something'] in wandb.init
        wandb_run = wandb.init(config=args.__dict__, reinit=True)
        wandb_run.name = args.work_dir_exp.split('/')[-1]  # run name
        
        args = update_args(args, wandb.config)
        # save args as yaml file every experiment
        yamldir = osp.join(os.getcwd(), args.work_dir_exp+'/train_config.yml')
        with open(yamldir, 'w') as f: yaml.dump(args.__dict__, f, indent=4)
        
        do_training(**args.__dict__)
        wandb_run.finish()
    else:
        # you must to change project name
        # if you want to use tags, put tags=['something'] in wandb.init
        # if you want to use group, put group='something' in wandb.init
        wandb.init(
            entity='mg_generation', project='data_annotation_baekkr',
            name=args.work_dir_exp.split('/')[-1],
            config=args.__dict__, reinit=True
        )
        
        # save args as yaml file every experiment
        yamldir = osp.join(os.getcwd(), args.work_dir_exp+'/train_config.yml')
        with open(yamldir, 'w') as f: yaml.dump(args.__dict__, f, indent=4)

        do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    if args.sweep:
        sweep_cfg = get_sweep_cfg()
        # you must to change project name
        sweep_id = wandb.sweep(sweep=sweep_cfg, entity='mg_generation', project='data_annotation_baekkr')
        wandb.agent(sweep_id=sweep_id, function=partial(main, args))
    else:
        main(args)
