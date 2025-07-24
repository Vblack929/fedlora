#!/bin/bash

# Example script showing how to run multiple federated learning experiments
# with different configurations

echo "Running multiple federated learning experiments..."
echo "=================================================="

# Experiment 1: Baseline with different attacker ratios
# echo "Experiment 1: Testing different attacker ratios with FedAvg"
# ./run_qwen_fl.sh --attackers 0.1 --poison_ratio 0.3 --epochs 3
# ./run_qwen_fl.sh --attackers 0.3 --poison_ratio 0.3 --epochs 3
# ./run_qwen_fl.sh --attackers 0.5 --poison_ratio 0.3 --epochs 3

# Experiment 2: Testing different defense methods
echo "Experiment 2: Testing different defense methods"
./run_qwen_fl.sh --defense ours --attackers 0.6 --poison_ratio 1.0 --epochs 3

# # Experiment 3: Testing different attack types
# echo "Experiment 3: Testing different attack types"
# ./run_qwen_fl.sh --attack_type addWord --attackers 0.3 --poison_ratio 0.5 --epochs 3
# ./run_qwen_fl.sh --attack_type addSent --attackers 0.3 --poison_ratio 0.5 --epochs 3
# ./run_qwen_fl.sh --attack_type lwp --attackers 0.3 --poison_ratio 0.5 --epochs 3

# # Experiment 4: Testing different LoRA ranks
# echo "Experiment 4: Testing different LoRA ranks"
# ./run_qwen_fl.sh --rank 4 --attackers 0.3 --poison_ratio 0.3 --epochs 3
# ./run_qwen_fl.sh --rank 8 --attackers 0.3 --poison_ratio 0.3 --epochs 3
# ./run_qwen_fl.sh --rank 16 --attackers 0.3 --poison_ratio 0.3 --epochs 3

echo "All experiments completed!" 