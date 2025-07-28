#!/bin/bash

# Comprehensive Federated Learning Experiments
# This script runs all experiments on a specified model (qwen or llama) 
# with different defense methods and attack types

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default parameters
MODEL="qwen"
DATASET="sst2"
NUM_USERS=10
EPOCHS=5
LOCAL_EPOCHS=5
FRAC=0.4
ATTACKERS=0.4
POISON_RATIO=1.0
RANK=8
LR=2e-4
DEVICE="auto"
SEED=""

# Experiment arrays
DEFENSE_METHODS=("fedavg" "krum" "multi_krum" "bulyan" "trimmed_mean" "ours")
ATTACK_TYPES=("addWord" "addSent" "lwp")
DATASETS=("sst2" "agnews")

# Function to display usage
usage() {
    echo -e "${CYAN}Usage: $0 [OPTIONS]${NC}"
    echo -e "${YELLOW}Options:${NC}"
    echo "  --model MODEL            Model name: qwen|llama (default: qwen)"
    echo "  --dataset DATASET        Dataset: sst2|agnews|both (default: sst2)"
    echo "  --num_users NUM          Number of users (default: 10)"
    echo "  --epochs EPOCHS          Number of training rounds (default: 5)"
    echo "  --local_epochs EPOCHS    Number of local epochs (default: 5)"
    echo "  --frac FRAC              Fraction of clients (default: 0.4)"
    echo "  --attackers ATTACKERS    Fraction of attackers (default: 0.4)"
    echo "  --poison_ratio RATIO     Poison ratio (default: 1.0)"
    echo "  --rank RANK              LoRA rank (default: 8)"
    echo "  --lr LR                  Learning rate (default: 2e-4)"
    echo "  --device DEVICE          Device: auto|cuda|mps|cpu (default: auto)"
    echo "  --seed SEED              Random seed (optional)"
    echo "  --quick                  Run quick experiments (only fedavg and addWord)"
    echo "  --defense-only DEFENSE   Run only specific defense method"
    echo "  --attack-only ATTACK     Run only specific attack type"
    echo "  --help                   Show this help message"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  # Run all experiments with qwen on sst2"
    echo "  $0 --model qwen --dataset sst2"
    echo ""
    echo "  # Run all experiments with llama on both datasets"
    echo "  $0 --model llama --dataset both"
    echo ""
    echo "  # Quick test with specific configuration"
    echo "  $0 --model qwen --quick --epochs 2"
    echo ""
    echo "  # Run only specific defense method"
    echo "  $0 --model qwen --defense-only ours"
    echo ""
    echo "  # Run only specific attack type"
    echo "  $0 --model llama --attack-only lwp"
}

# Parse command line arguments
QUICK_MODE=false
DEFENSE_ONLY=""
ATTACK_ONLY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            if [[ "$MODEL" != "qwen" && "$MODEL" != "llama" ]]; then
                echo -e "${RED}Error: Model must be 'qwen' or 'llama'${NC}"
                exit 1
            fi
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            if [[ "$DATASET" != "sst2" && "$DATASET" != "agnews" && "$DATASET" != "both" ]]; then
                echo -e "${RED}Error: Dataset must be 'sst2', 'agnews', or 'both'${NC}"
                exit 1
            fi
            shift 2
            ;;
        --num_users)
            NUM_USERS="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --local_epochs)
            LOCAL_EPOCHS="$2"
            shift 2
            ;;
        --frac)
            FRAC="$2"
            shift 2
            ;;
        --attackers)
            ATTACKERS="$2"
            shift 2
            ;;
        --poison_ratio)
            POISON_RATIO="$2"
            shift 2
            ;;
        --rank)
            RANK="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --defense-only)
            DEFENSE_ONLY="$2"
            shift 2
            ;;
        --attack-only)
            ATTACK_ONLY="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Adjust arrays based on options
if [[ "$QUICK_MODE" == true ]]; then
    DEFENSE_METHODS=("fedavg")
    ATTACK_TYPES=("addWord")
    echo -e "${YELLOW}Running in quick mode: only fedavg + addWord${NC}"
fi

if [[ -n "$DEFENSE_ONLY" ]]; then
    DEFENSE_METHODS=("$DEFENSE_ONLY")
    echo -e "${YELLOW}Running only defense method: $DEFENSE_ONLY${NC}"
fi

if [[ -n "$ATTACK_ONLY" ]]; then
    ATTACK_TYPES=("$ATTACK_ONLY")
    echo -e "${YELLOW}Running only attack type: $ATTACK_ONLY${NC}"
fi

# Set datasets to run
if [[ "$DATASET" == "both" ]]; then
    DATASETS_TO_RUN=("sst2" "agnews")
else
    DATASETS_TO_RUN=("$DATASET")
fi

# Function to run a single experiment
run_experiment() {
    local model=$1
    local dataset=$2
    local defense=$3
    local attack=$4
    local exp_num=$5
    local total_exp=$6
    
    echo -e "${PURPLE}[$exp_num/$total_exp] Running: Model=$model, Dataset=$dataset, Defense=$defense, Attack=$attack${NC}"
    
    # Determine which script to use based on dataset
    local script_name
    local python_script
    if [[ "$dataset" == "sst2" ]]; then
        python_script="qwen_lora_sst2.py"
    else
        python_script="qwen_lora_agnews.py"
    fi
    
    # Construct the command
    local cmd="python $python_script"
    cmd="$cmd --model $model"
    cmd="$cmd --dataset $dataset"
    cmd="$cmd --num_users $NUM_USERS"
    cmd="$cmd --epochs $EPOCHS"
    
    # Add local_epochs only for agnews (based on existing scripts)
    if [[ "$dataset" == "agnews" ]]; then
        cmd="$cmd --local_epochs $LOCAL_EPOCHS"
    fi
    
    cmd="$cmd --frac $FRAC"
    cmd="$cmd --attackers $ATTACKERS"
    cmd="$cmd --poison_ratio $POISON_RATIO"
    cmd="$cmd --attack_type $attack"
    cmd="$cmd --defense $defense"
    cmd="$cmd --rank $RANK"
    cmd="$cmd --lr $LR"
    cmd="$cmd --device $DEVICE"
    
    if [[ -n "$SEED" ]]; then
        cmd="$cmd --seed $SEED"
    fi
    
    # Create log directory if it doesn't exist
    mkdir -p logs
    
    # Create log filename
    local log_file="logs/exp_${model}_${dataset}_${defense}_${attack}_$(date +%Y%m%d_%H%M%S).log"
    
    echo -e "${CYAN}Command: $cmd${NC}"
    echo -e "${CYAN}Log file: $log_file${NC}"
    
    # Run the experiment and capture output
    if eval "$cmd" 2>&1 | tee "$log_file"; then
        echo -e "${GREEN}✓ Experiment completed successfully${NC}"
        echo "Results logged to: $log_file"
    else
        echo -e "${RED}✗ Experiment failed${NC}"
        echo "Error details in: $log_file"
    fi
    
    echo ""
}

# Calculate total number of experiments
total_experiments=$((${#DATASETS_TO_RUN[@]} * ${#DEFENSE_METHODS[@]} * ${#ATTACK_TYPES[@]}))

# Display experiment plan
echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}           FEDERATED LEARNING EXPERIMENT SUITE${NC}"
echo -e "${BLUE}================================================================${NC}"
echo -e "${YELLOW}Configuration:${NC}"
echo "Model: $MODEL"
echo "Datasets: ${DATASETS_TO_RUN[*]}"
echo "Defense methods: ${DEFENSE_METHODS[*]}"
echo "Attack types: ${ATTACK_TYPES[*]}"
echo "Number of users: $NUM_USERS"
echo "Epochs: $EPOCHS"
echo "Local epochs: $LOCAL_EPOCHS"
echo "Client fraction: $FRAC"
echo "Attacker fraction: $ATTACKERS"
echo "Poison ratio: $POISON_RATIO"
echo "LoRA rank: $RANK"
echo "Learning rate: $LR"
echo "Device: $DEVICE"
if [[ -n "$SEED" ]]; then
    echo "Seed: $SEED"
fi
echo ""
echo -e "${YELLOW}Total experiments to run: $total_experiments${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

# Confirm before running (unless in quick mode)
if [[ "$QUICK_MODE" != true ]]; then
    read -p "Do you want to proceed with all experiments? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Experiments cancelled."
        exit 0
    fi
fi

# Record start time
start_time=$(date)
echo -e "${GREEN}Starting experiments at: $start_time${NC}"
echo ""

# Run all experiments
experiment_counter=0
failed_experiments=0

for dataset in "${DATASETS_TO_RUN[@]}"; do
    for defense in "${DEFENSE_METHODS[@]}"; do
        for attack in "${ATTACK_TYPES[@]}"; do
            experiment_counter=$((experiment_counter + 1))
            
            if ! run_experiment "$MODEL" "$dataset" "$defense" "$attack" "$experiment_counter" "$total_experiments"; then
                failed_experiments=$((failed_experiments + 1))
            fi
            
            # Small delay between experiments
            sleep 2
        done
    done
done

# Summary
end_time=$(date)
echo -e "${BLUE}================================================================${NC}"
echo -e "${GREEN}All experiments completed!${NC}"
echo -e "${YELLOW}Summary:${NC}"
echo "Started: $start_time"
echo "Finished: $end_time"
echo "Total experiments: $total_experiments"
echo "Successful: $((total_experiments - failed_experiments))"
if [[ $failed_experiments -gt 0 ]]; then
    echo -e "${RED}Failed: $failed_experiments${NC}"
else
    echo -e "${GREEN}Failed: 0${NC}"
fi
echo ""
echo "Log files are stored in the 'logs/' directory"
echo "Results are typically saved in 'pilot/' or 'results/' directories"
echo -e "${BLUE}================================================================${NC}" 