import torch
import torch.nn.functional as F

from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc

from .modeling_vit_seq import ViTForSemanticSegmentation_seq
from .configuration_vit import ViTConfig
from mmengine.model import BaseModel

class MMViT_seq(BaseModel):
    def __init__(self, img_size):
        super().__init__()
        # name = 'glasses/vit_huge_patch32_384'
        name = 'google/vit-base-patch16-224'
        model_config = ViTConfig.from_pretrained(name, num_labels=24, image_size=img_size, output_hidden_states=True)
        self.vit = ViTForSemanticSegmentation_seq.from_pretrained(name, config=model_config, ignore_mismatched_sizes=True)
        
    def forward(self, imgs, data_samples=None, mode='tensor'):
        label = data_samples['labels']
        imgs = imgs.contiguous()
        label = label.contiguous()
        torch.distributed.broadcast(imgs, src=0)
        torch.distributed.broadcast(label, src=0)
        imgs = torch.chunk(imgs, gpc.get_world_size(ParallelMode.SEQUENCE), dim=-2)[gpc.get_local_rank(ParallelMode.SEQUENCE)]
        label = torch.chunk(label, gpc.get_world_size(ParallelMode.SEQUENCE), dim=-2)[gpc.get_local_rank(ParallelMode.SEQUENCE)]
        data_samples['labels'] = label
        
        # with torch.autograd.graph.save_on_cpu(pin_memory=True):
        out = self.vit(pixel_values=imgs, labels=label)
        if mode == 'loss':
            loss = F.cross_entropy(out[0].float(), label, ignore_index=255)
            return {'loss': loss}
        elif mode == 'predict':
            return out[0], data_samples