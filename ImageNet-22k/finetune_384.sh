MODEL=van_b4
DROP_PATH=0.2 # 0.2 for b4, b5; 0.3 for b6
DATA_PATH=/path/to/1k
PRETRAIN=/path/to/pretrain_model
python3 -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=9001 --nnodes=4 --node_rank={YOUR_NODE_ID} --master_addr={MASTER_ADDR} \
    main.py \
    --model ${MODEL} --drop_path ${DROP_PATH} --input_size 384 \
    --batch_size 16 --lr 5e-5 --update_freq 1 \
    --warmup_epochs 0 --epochs 30 --weight_decay 1e-8 \
    --head_init_scale 0.001 --cutmix 0 --mixup 0 \
    --finetune ${PRETRAIN} \
    --data_path ${DATA_PATH} \
