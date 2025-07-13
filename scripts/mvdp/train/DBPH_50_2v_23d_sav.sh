accelerate launch --num_processes 8\
    --config_file configs/accelerate_config.yaml \
    main.py \
    --config-path=configs/mvdp_train \
    --config-name=mvdp_2v_23d.yaml \
    train_dataset.task=dual_bottles_pick_hard \
    train_dataset.demo=50