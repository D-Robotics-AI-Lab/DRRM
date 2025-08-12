config=mvdp4thanos_1v_23d_23_global
task='Place the blue square object into the bowl'
demo=None
path="${config/_*/}_${task}_${demo}_${config#*_}"

accelerate launch\
    --config_file configs/accelerate_config.yaml \
    main.py \
    --config-path=configs/mvdp_train \
    --config-name=$config.yaml \
    train_dataset.task="$task" \
    train_dataset.demo=$demo

~/bcecmd bos cp -r "checkpoints/$path" "bos:/dg-algo/zehao.ni/checkpoints/$path"