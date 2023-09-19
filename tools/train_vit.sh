# CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=2 test_mmeng_fbp_colo.py --launcher pytorch --cfg swin/configs/swin_large_patch4_window12_my.yaml

python -m torch.distributed.launch --nproc_per_node=2 tools/train_vit.py --launcher pytorch

# torchrun --nproc_per_node=4 test_mmeng_fbp_colo.py --cfg swin/configs/swin_large_patch4_window12_my.yaml
