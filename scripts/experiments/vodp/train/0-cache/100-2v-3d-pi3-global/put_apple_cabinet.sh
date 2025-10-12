config=mvdp_2v_3d_pi3_global
task=put_apple_cabinet
demo=100
path="${config/_*/}_${task}_${demo}_${config#*_}"

accelerate launch\
    --config_file configs/accelerate_config.yaml \
    main.py \
    --config-path=configs/mvdp_train \
    --config-name=$config.yaml \
    train_dataset.task=$task \
    train_dataset.demo=$demo

~/bcecmd bos cp -r checkpoints/$path bos:/dg-algo/zehao.ni/checkpoints/$path