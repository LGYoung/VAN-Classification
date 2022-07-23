MODEL=van_b4
DROP_PATH=0.1
DATA_PATH=/path/to/22k
python3 -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=9001 --nnodes=8 --node_rank={YOUR_NODE_ID} --master_addr={MASTER_ADDR} \
    main.py \
    --model ${MODEL} --drop_path ${DROP_PATH} \
    --batch_size 64 --lr 4e-3 --update_freq 2 \
    --warmup_epochs 5 --epochs 90 \
    --data_set image_folder --nb_classes 21841 --disable_eval true \
    --data_path ${DATA_PATH} \