import os
import csv
import os.path as osp
import numpy as np
import shutil
import cv2
import torch

from mmengine.dist import master_only
from mmengine.hooks import Hook

def create_palette(csv_filepath):
    color_to_class = {}
    with open(csv_filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            r, g, b = int(row['r']), int(row['g']), int(row['b'])
            color_to_class[(r, g, b)] = idx
    return color_to_class

class SegVisHook(Hook):

    def __init__(self, data_root, vis_num=1) -> None:
        super().__init__()
        self.vis_num = vis_num
        self.palette = create_palette(osp.join(data_root, 'class_dict.csv'))

    @master_only
    def after_val_iter(self,
                       runner,
                       batch_idx: int,
                       data_batch=None,
                       outputs=None) -> None:
        if batch_idx > self.vis_num:
            return

        preds, data_samples = outputs
        img_paths = data_samples['img_path']
        mask_paths = data_samples['mask_path']
        _, C, H, W = preds.shape
        preds = torch.argmax(preds, dim=1)
        for idx, (pred, img_path,
                  mask_path) in enumerate(zip(preds, img_paths, mask_paths)):
            pred_mask = np.zeros((H, W, 3), dtype=np.uint8)
            runner.visualizer.set_image(pred_mask)
            for color, class_id in self.palette.items():
                runner.visualizer.draw_binary_masks(
                    pred == class_id,
                    colors=[color],
                    alphas=1.0,
                )
            # Convert RGB to BGR
            pred_mask = runner.visualizer.get_image()[..., ::-1]
            saved_dir = osp.join(runner.log_dir, 'vis_data', str(idx))
            os.makedirs(saved_dir, exist_ok=True)

            shutil.copyfile(img_path,
                            osp.join(saved_dir, osp.basename(img_path)))
            shutil.copyfile(mask_path.replace('__index','__color'),
                            osp.join(saved_dir, osp.basename(mask_path)))
            cv2.imwrite(
                osp.join(saved_dir, f'pred_{osp.basename(mask_path)}'),
                pred_mask)
