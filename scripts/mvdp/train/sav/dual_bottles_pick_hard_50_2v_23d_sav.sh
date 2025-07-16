config=mvdp_2v_23d
task=dual_bottles_pick_hard
demo=50
path="${config/_*/}_${task}_${demo}_${config#*_}_sav"

accelerate launch --num_processes 8\
    --config_file configs/accelerate_config.yaml \
    main.py \
    --config-path=configs/mvdp_train \
    --config-name=$config.yaml \
    model.obs_encoder.emvis_config.seq_as_view=True \
    train_dataset.task=$task \
    train_dataset.demo=$demo

~/bcecmd bos cp -r checkpoints/$path bos:/dg-algo/zehao.ni/checkpoints/$path