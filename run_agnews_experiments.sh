#!/bin/bash

# AG News Federated Learning Experiments
# This script runs multiple experiments with different parameter combinations

echo "Starting AG News Federated Learning Experiments..."
echo "=================================================="

# Parameter arrays - uncomment/modify as needed
defense_methods=("fedavg" "krum" "multi_krum" "bulyan" "trimmed_mean")
attack_types=("addWord" "addSent" "lwp") 
poison_ratios=(0.3 0.5 1.0)
attackers=(0.1 0.3 0.5)
ranks=(8 16)
learning_rates=(2e-4)
epochs=5
local_epochs=5
num_users=20
frac=0.4

# Experiment 1: Test different defense methods with standard parameters
echo "Experiment 1: Testing different defense methods"
echo "----------------------------------------------"
for defense in "${defense_methods[@]}"
do
    echo "Running defense: $defense"
    ./run_qwen_agnews.sh \
        --defense $defense \
        --attack_type addWord \
        --attackers 0.4 \
        --poison_ratio 1.0 \
        --epochs $epochs \
        --local_epochs $local_epochs \
        --rank 8 \
        --lr 2e-4
done

# echo ""
# echo "Experiment 2: Testing different attack types"
# echo "-------------------------------------------"
# for attack_type in "${attack_types[@]}"
# do
#     echo "Running attack type: $attack_type"
#     ./run_qwen_agnews.sh \
#         --defense fedavg \
#         --attack_type $attack_type \
#         --attackers 0.3 \
#         --poison_ratio 0.5 \
#         --epochs $epochs \
#         --local_epochs $local_epochs \
#         --rank 8 \
#         --lr 2e-4
# done

# echo ""
# echo "Experiment 3: Testing different attacker ratios"
# echo "----------------------------------------------"
# for attacker in "${attackers[@]}"
# do
#     echo "Running with attacker ratio: $attacker"
#     ./run_qwen_agnews.sh \
#         --defense fedavg \
#         --attack_type addWord \
#         --attackers $attacker \
#         --poison_ratio 0.5 \
#         --epochs $epochs \
#         --local_epochs $local_epochs \
#         --rank 8 \
#         --lr 2e-4
# done

# echo ""
# echo "Experiment 4: Testing different poison ratios"
# echo "--------------------------------------------"
# for poison_ratio in "${poison_ratios[@]}"
# do
#     echo "Running with poison ratio: $poison_ratio"
#     ./run_qwen_agnews.sh \
#         --defense fedavg \
#         --attack_type addWord \
#         --attackers 0.3 \
#         --poison_ratio $poison_ratio \
#         --epochs $epochs \
#         --local_epochs $local_epochs \
#         --rank 8 \
#         --lr 2e-4
# done

# echo ""
# echo "Experiment 5: Testing different LoRA ranks"
# echo "-----------------------------------------"
# for rank in "${ranks[@]}"
# do
#     echo "Running with LoRA rank: $rank"
#     ./run_qwen_agnews.sh \
#         --defense fedavg \
#         --attack_type addWord \
#         --attackers 0.3 \
#         --poison_ratio 0.5 \
#         --epochs $epochs \
#         --local_epochs $local_epochs \
#         --rank $rank \
#         --lr 2e-4
done

# Uncomment this section for comprehensive grid search
# echo ""
# echo "Comprehensive Grid Search (Warning: This will run many experiments)"
# echo "=================================================================="
# read -p "Do you want to run comprehensive experiments? (y/n): " -n 1 -r
# echo
# if [[ $REPLY =~ ^[Yy]$ ]]
# then
#     echo "Running comprehensive experiments..."
#     counter=0
#     total=$((${#defense_methods[@]} * ${#attack_types[@]} * ${#poison_ratios[@]} * ${#attackers[@]} * ${#ranks[@]}))
#     
#     for defense in "${defense_methods[@]}"
#     do
#         for attack_type in "${attack_types[@]}"
#         do
#             for poison_ratio in "${poison_ratios[@]}"
#             do
#                 for attacker in "${attackers[@]}"
#                 do
#                     for rank in "${ranks[@]}"
#                     do
#                         counter=$((counter + 1))
#                         echo "[$counter/$total] Running: $defense, $attack_type, $poison_ratio, $attacker, $rank"
#                         ./run_qwen_agnews.sh \
#                             --defense $defense \
#                             --attack_type $attack_type \
#                             --attackers $attacker \
#                             --poison_ratio $poison_ratio \
#                             --epochs $epochs \
#                             --local_epochs $local_epochs \
#                             --rank $rank \
#                             --lr 2e-4
#                     done
#                 done
#             done
#         done
#     done
# fi

echo ""
echo "All AG News experiments completed!"
echo "Results saved in pilot/ directory" 