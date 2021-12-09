python -m torch.distributed.launch --nproc_per_node=4 --master_port=8340  --use_env main_clean.py --model resnet50 --batch-size 64 --data-path /mnt/sdb/meijieru/imagenet  --epoch 100  --reprob 0  --no-repeated-aug --drop 0 --drop-path 0 --warmup-epochs 5 --adjust_lr 256  --aa 'rand-m9-mstd0.5-inc1'  --sing singbn