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
        
    def forward(self, inputs, data_samples=None, mode='tensor'):      
        img = [x.unsqueeze(0) for x in inputs]
        img = torch.cat(img, 0)
        label = [(y.gt_sem_seg.data.type(torch.uint8) - 1).type(torch.long) for y in data_samples]
        label = torch.cat(label, 0)
        
        img = img.contiguous()
        label = label.contiguous()
        torch.distributed.broadcast(img, src=0)
        torch.distributed.broadcast(label, src=0)
        img = torch.chunk(img, gpc.get_world_size(ParallelMode.SEQUENCE), dim=-2)[gpc.get_local_rank(ParallelMode.SEQUENCE)]
        label = torch.chunk(label, gpc.get_world_size(ParallelMode.SEQUENCE), dim=-2)[gpc.get_local_rank(ParallelMode.SEQUENCE)]
        # for i in range(len(data_samples)):
        #     data_samples[i].gt_sem_seg.data = label[i]
        
        # with torch.autograd.graph.save_on_cpu(pin_memory=True):
        out = self.vit(pixel_values=img, labels=label)
        if mode == 'loss':
            loss = F.cross_entropy(out[0].float(), label, ignore_index=255)
            return {'loss': loss}
        elif mode == 'predict':
            return out[0], data_samples