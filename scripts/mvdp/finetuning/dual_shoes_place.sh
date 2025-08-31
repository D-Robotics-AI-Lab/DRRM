config=mvdp_1v_23d_23_pool
task=dual_shoes_place
path="${config/_*/}_${task}_ft_${config#*_}"

accelerate launch\
    --config_file configs/accelerate_config.yaml \
    main.py \
    --config-path=configs/mvdp_finetuning \
    --config-name=$config.yaml \
    train_dataset.task=$task

~/bcecmd bos cp -r checkpoints/$path bos:/dg-algo/zehao.ni/checkpoints/$path