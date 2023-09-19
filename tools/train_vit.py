import argparse
import numpy as np

import torch
import torchvision.transforms as transforms

import colossalai

from datasets import FBP
from models import MMViT_seq
from utils import SegVisHook

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
        default=512,
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

    norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    transform = transforms.Compose(
                [
                transforms.ToTensor(),
                transforms.Normalize(**norm_cfg),
                transforms.CenterCrop(args.img_size),
                ])

    target_transform = transforms.Compose(
                [
                transforms.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.long)),
                transforms.CenterCrop(args.img_size),
                ])

    train_set = FBP(
        '/media/dell/data1/cw/data/Five-Billion-Pixels/fbp_2048/',
        img_folder='Image_RGB/test',
        mask_folder='Annotation__index/test',
        transform=transform,
        target_transform=target_transform)

    valid_set = FBP(
        '/media/dell/data1/cw/data/Five-Billion-Pixels/fbp_2048/',
        img_folder='Image_RGB/val',
        mask_folder='Annotation__index/val',
        transform=transform,
        target_transform=target_transform)

    train_dataloader = dict(
        batch_size=args.batch_size,
        dataset=train_set,
        sampler=dict(type='DefaultSampler', shuffle=True),
        collate_fn=dict(type='default_collate'))

    val_dataloader = dict(
        batch_size=args.batch_size,
        dataset=valid_set,
        sampler=dict(type='DefaultSampler', shuffle=False),
        collate_fn=dict(type='default_collate'))
    
    model = MMViT_seq(img_size=args.img_size)
    
    from mmengine.runner._flexible_runner import FlexibleRunner
    from colossalai.tensor.op_wrapper import colo_op_impl

    colo_op_impl(torch.Tensor.add_)(torch.add)
    strategy = dict(type='ColossalAIStrategy', mixed_precision='fp16', plugin='lowlevel-zero')
    optim_wrapper = dict(optimizer=dict(type='HybridAdam', lr=1e-4))
    
    from mmseg.evaluation import IoUMetric
    runner = FlexibleRunner(
        model=model,
        work_dir='./work_dirs/fbp_vit_seq',
        strategy=strategy,
        train_dataloader=train_dataloader,
        optim_wrapper=optim_wrapper,
        param_scheduler=dict(type='LinearLR'),
        train_cfg=dict(by_epoch=True, max_epochs=50, val_interval=1),
        val_dataloader=val_dataloader,
        val_cfg=dict(),
        val_evaluator=dict(type=IoUMetric),
        launcher=args.launcher,
        custom_hooks=[SegVisHook('/media/dell/data1/cw/data/Five-Billion-Pixels/fbp_2048', vis_num=10)],
        default_hooks=dict(checkpoint=dict(type='CheckpointHook', interval=1)),
        visualizer=dict(type='Visualizer', vis_backends=[dict(type='WandbVisBackend')]),
    )
       
    runner.train()

if __name__ == '__main__':
    main()