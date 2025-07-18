config=mvdp_2v_3d
task=bottle_adjust
demo=100
path="${config/_*/}_${task}_${demo}_${config#*_}"

accelerate launch --num_processes 8\
    --config_file configs/accelerate_config.yaml \
    main.py \
    --config-path=configs/mvdp_train \
    --config-name=$config.yaml \
    train_dataset.task=$task \
    train_dataset.demo=$demo

~/bcecmd bos cp -r checkpoints/$path bos:/dg-algo/zehao.ni/checkpoints/$path