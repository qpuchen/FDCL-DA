export tag='gbl_sb7_fullb_1111_np2_lr1e-3_mul5_bs12_448_ep200_linear_oricon'
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12346  main.py \
--dataset soybean2000 \
--tag $tag \
--lr 1e-3 \
--model full \
--mask \
--swap \
--con \
--use_selection \
--margin 1 \
--origin_w 1 \
--swap_w 1 \
--con_w 1.25 \
--fd_w 1 \
--num_part 2 \
--img_size 448 \
--cfg configs/swin/swin_base_patch4_window7_448.yaml \
--data-path ./datasets \
--batch-size 12 \
--eval-batch-size 12 \
--epochs 200 \
--pretrained ./pretrained_models/swin_base_patch4_window7_224_22k.pth \
>> logs/soygbl/$tag 2>&1 & \
tail -fn 50 logs/soygbl/$tag

