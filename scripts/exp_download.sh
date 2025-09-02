task_list=('block_hammer_beat' 'put_apple_cabinet' 'dual_bottles_pick_hard' 'bottle_adjust'
            'container_place')
            
for task in "${task_list[@]}"; do
    # domain=dp_${task}_100_baseline
    # ~/bcecmd bos cp -r bos:/dg-algo/zehao.ni/eval_result/${domain}/exp1/ eval_result/${domain}/exp1/
    # domain=mvdp_${task}_100_1v_3d_23_frame
    # ~/bcecmd bos cp -r bos:/dg-algo/zehao.ni/eval_result/${domain}/exp1/ eval_result/${domain}/exp1/
    # domain=mvdp_${task}_100_1v_3d_23_global
    # ~/bcecmd bos cp -r bos:/dg-algo/zehao.ni/eval_result/${domain}/exp1/ eval_result/${domain}/exp1/
    # domain=mvdp_${task}_100_1v_23d_23_global
    # ~/bcecmd bos cp -r bos:/dg-algo/zehao.ni/eval_result/${domain}/exp1/ eval_result/${domain}/exp1/
    # domain=mvdp_${task}_100_1v_32d_23_global
    # ~/bcecmd bos cp -r bos:/dg-algo/zehao.ni/eval_result/${domain}/exp1/ eval_result/${domain}/exp1/
    # domain=mvdp_${task}_100_1v_2d
    # ~/bcecmd bos cp -r bos:/dg-algo/zehao.ni/eval_result/${domain}/exp1/ eval_result/${domain}/exp1/
done

domain=mvdp_container_place_100_1v_23d_23_pool
~/bcecmd bos cp -r bos:/dg-algo/zehao.ni/eval_result/${domain}/exp1/ eval_result/${domain}/exp1