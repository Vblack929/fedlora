#!/bin/bash

# Array of defense methods to run
# defense_methods=("fedavg" "trimmedmean" "krum" "multi_krum" "bulyan")
# repeat_times=(1)
defense_methods=("fedavg") 
attack_types=("addWord") 
poison_ratios=(0.7)
learning_rates=(1e-4)
attackers=(0.4)
rank=(4 8 16 32)

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
          for attacker in "${attackers[@]}"
          do
            for r in "${rank[@]}"
            do
              echo "Running with attack: $attack_type, poison ratio: $poison_ratio, lr: $learning_rate, attacker: $attacker, rank: $r"
              python main.py \
                --model distilbert \
                --epochs 5 \
                --local_ep 5 \
                --local_bs 32 \
              --dataset sst2 \
            --tuning lora \
            --num_classes 2 \
            --num_users 30 \
            --frac 0.4 \
            --attackers $attacker \
            --attack_type $attack_type \
            --lr $learning_rate \
            --optimizer adamw \
            --gpu \
            --defense $defense \
            --poison_ratio $poison_ratio \
            --rank $r
            done
          done
        done
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