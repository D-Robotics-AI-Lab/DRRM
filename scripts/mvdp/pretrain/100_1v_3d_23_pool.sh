config=mvdp_1v_3d_23_pool
path="${config/_*/}_pretrain_${config#*_}"

accelerate launch\
    --config_file configs/accelerate_config_mm.yaml \
    --machine_rank ${DEEPSEED_RANK} \
    --main_process_ip 192.168.80.7 \
    --main_process_ip 10086 \
    main.py \
    --config-path=configs/mvdp_pretrain \
    --config-name=$config.yaml \

~/bcecmd bos cp -r checkpoints/$path bos:/dg-algo/zehao.ni/checkpoints/$path