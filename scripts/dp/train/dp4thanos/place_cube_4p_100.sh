config=dp_baseline
task=place_cube_4p_100
demo=null
path="${config/_*/}_${task}_None_${config#*_}"

accelerate launch\
    --config_file configs/accelerate_config.yaml \
    main.py \
    --config-path=configs/mvdp_train \
    --config-name=$config.yaml \
    output_dir=checkpoints/$path \
    train_dataset.task=$task \
    train_dataset.demo=$demo \
    train_dataset.root=datasets/thanos/$task \
    train_dataset.repo_id=D-robotics/$task

~/bcecmd bos cp -r "checkpoints/$path" "bos:/dg-algo/zehao.ni/checkpoints/$path"