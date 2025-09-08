config=mvdp_1v_23d_23_pool_single_frame
task=place_tiny_cube_200
demo=null
path="${config/_*/}_${task}_None_${config#*_}"

accelerate launch\
    --config_file configs/accelerate_config.yaml \
    main.py \
    --config-path=configs/mvdp_train \
    --config-name=$config.yaml \
    train_dataset.task=$task \
    train_dataset.demo=$demo

~/bcecmd bos cp -r "checkpoints/$path" "bos:/dg-algo/zehao.ni/checkpoints/$path"