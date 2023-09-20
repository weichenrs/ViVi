import argparse
import numpy as np

import torch
import torchvision.transforms as transforms

import colossalai

from datasets import FBPDataset
from models import MMViT_seq
from utils import SegVisHook

from mmseg.datasets import * #RandomCrop、PackSegInputs
from mmseg.evaluation import * #IoUMetric
from mmseg.engine import * #SegVisualizationHook

def parse_option():
    parser = argparse.ArgumentParser('ViT training and evaluation script', add_help=False)
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=1,
        help='input batch size')
    parser.add_argument(
        '--img_size', 
        type=int, 
        default=2048,
        help='input image size')
    parser.add_argument(
        '--colocfg', 
        type=str, 
        default='configs/colo_config.py',
        help='path to the config file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    args, _ = parser.parse_known_args()
    return args

def main():
    args = parse_option()
    colossalai.launch_from_torch(config=args.colocfg, seed=4396)

    dataset_type = 'FBPDataset'
    data_root = '/media/dell/data1/cw/data/Five-Billion-Pixels/fbp_2048/'
    crop_size = (2048, 2048)
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        # dict(
        #     type='RandomResize',
        #     scale=(2048, 2048),
        #     ratio_range=(1.0, 2.0),
        #     keep_ratio=True),
        # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='PackSegInputs')
    ]
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        # dict(type='Resize', scale=(2048, 2048), keep_ratio=True),
        # add loading annotation after ``Resize`` because ground truth
        # does not need to do resize data transform
        dict(type='LoadAnnotations'),
        dict(type='PackSegInputs')
    ]
    img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    tta_pipeline = [
        dict(type='LoadImageFromFile', backend_args=None),
        dict(
            type='TestTimeAug',
            transforms=[
                [
                    dict(type='Resize', scale_factor=r, keep_ratio=True)
                    for r in img_ratios
                ],
                [
                    dict(type='RandomFlip', prob=0., direction='horizontal'),
                    dict(type='RandomFlip', prob=1., direction='horizontal')
                ], 
                [dict(type='LoadAnnotations')], 
                [dict(type='PackSegInputs')]
            ])
    ]
    train_dataloader = dict(
        batch_size=1,
        num_workers=2,
        persistent_workers=True,
        sampler=dict(type='InfiniteSampler', shuffle=True),
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(
                img_path='Image_RGB/train', 
                seg_map_path='Annotation__index/train'),
            pipeline=train_pipeline))
    val_dataloader = dict(
        batch_size=1,
        num_workers=4,
        persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(
                img_path='Image_RGB/val', 
                seg_map_path='Annotation__index/val'),
            pipeline=test_pipeline))
    test_dataloader = dict(
        batch_size=1,
        num_workers=4,
        persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(
                img_path='Image_RGB/test', 
                seg_map_path='Annotation__index/test'),
            pipeline=test_pipeline))

    val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
    test_evaluator = val_evaluator
    
    model = MMViT_seq(img_size=args.img_size)
    
    from mmengine.runner._flexible_runner import FlexibleRunner
    from colossalai.tensor.op_wrapper import colo_op_impl

    colo_op_impl(torch.Tensor.add_)(torch.add)
    strategy = dict(type='ColossalAIStrategy', mixed_precision='fp16', plugin='lowlevel-zero')
    optim_wrapper = dict(optimizer=dict(type='HybridAdam', lr=1e-4))

    runner = FlexibleRunner(               
        model=model,
        strategy=strategy,
        work_dir='./work_dirs/fbp_vit_seq',
        experiment_name='test_230919',
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=1000),
        val_cfg = dict(type='ValLoop'),
        test_cfg = dict(type='TestLoop'),
        val_evaluator=val_evaluator,
        test_evaluator=test_evaluator,
        # auto_scale_lr=dict(base_batch_size=16, enable=False),
        optim_wrapper=optim_wrapper,
        param_scheduler=dict(type='LinearLR'),
        default_hooks = dict(
            timer=dict(type='IterTimerHook'),
            logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
            param_scheduler=dict(type='ParamSchedulerHook'),
            checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000),
            sampler_seed=dict(type='DistSamplerSeedHook'),
            # visualization=dict(type='SegVisualizationHook')
            ),
        launcher=args.launcher,
        custom_hooks=[SegVisHook('/media/dell/data1/cw/data/Five-Billion-Pixels/fbp_2048', vis_num=5)],
        visualizer=dict(type='Visualizer', vis_backends=[dict(type='WandbVisBackend')]),
    )
       
    runner.train()

if __name__ == '__main__':
    main()