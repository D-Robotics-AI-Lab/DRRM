config=mvdp4thanos_1v_23d_23_pool
task=container_pick_blue
demo=null
path="${config/_*/}_${task}_None_${config#*_}"

accelerate launch\
    --config_file configs/accelerate_config.yaml \
    main.py \
    --config-path=configs/mvdp_train \
    --config-name=$config.yaml \
    train_dataset.task=$task

~/bcecmd bos cp -r "checkpoints/$path" "bos:/dg-algo/zehao.ni/checkpoints/$path"