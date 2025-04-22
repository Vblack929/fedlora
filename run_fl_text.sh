#!/bin/bash

# Array of defense methods to run
# defense_methods=("fedavg" "trimmedmean" "krum" "multi_krum" "bulyan")
# repeat_times=(1)
defense_methods=("krum") 
attack_types=("addWord") 
poison_ratios=(0.6)
learning_rates=(1e-4 3e-5 5e-5)

# Loop through each defense method
for defense in "${defense_methods[@]}"
do
    echo "Starting runs with defense method: $defense"
    for attack_type in "${attack_types[@]}"
    do
      for poison_ratio in "${poison_ratios[@]}"
      do
        for learning_rate in "${learning_rates[@]}"
        do
          echo "Running with attack: $attack_type, poison ratio: $poison_ratio, lr: $learning_rate"
          python main.py \
            --model distilbert \
            --epochs 5 \
            --local_ep 5 \
            --local_bs 32 \
            --dataset agnews \
            --tuning lora \
            --num_classes 2 \
            --num_users 30 \
            --frac 0.4 \
            --attackers 0.5 \
            --attack_type $attack_type \
            --lr $learning_rate \
            --optimizer adamw \
            --gpu \
            --defense $defense \
            --poison_ratio $poison_ratio
        done
      echo "---------------------------------------------"
      echo "Completed run with attack type: $attack_type"
      echo "---------------------------------------------"
    done
    echo "---------------------------------------------"
    echo "Completed all runs with defense method: $defense"
    echo "---------------------------------------------"
  done
done

echo "All experiments completed"