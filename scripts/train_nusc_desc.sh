python train_feats.py --batch_size 8 --epochs 100 --lr 0.001 --seed 1 --gpu GPU \
--npoints 8192 --dataset nuscenes --voxel_size 0.3 --ckpt_dir CKPT_DIR \
--use_fps --use_weights --data_list ./data/nuscenes_list --runname RUNNAME --augment 0.5 \
--root DATA_ROOT --wandb_dir WANDB_DIR --train_desc --freeze_detector \
--pretrain_detector PRETRAIN_DETECTOR --use_wandb