#!/bin/sh

env="BOSCH"
algo="rmappo"  # or "mappo"
exp="bosch_check"
seed_max=1

echo "env is ${env}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_bosch.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --seed ${seed} --num_lines 6 --num_products 3 --num_periods 24 \
    --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --ppo_epoch 15 \
    --capacity_per_line 100.0 --max_lot_size 10 --holding_cost 1.0 --backlog_cost 10.0 \
    --production_cost 1.0 --setup_cost 2.0 --pm_cost 20.0 --cm_cost 40.0 \
    --pm_time 2.0 --cm_time 4.0 --processing_time 1.0,1.0,1.0 --mean_demand 10.0,10.0,10.0 \
    --alpha_cost_weight 0.1 --hazard_rate 0.001 --max_actions_per_period 8 --use_wandb False --share_policy
done

