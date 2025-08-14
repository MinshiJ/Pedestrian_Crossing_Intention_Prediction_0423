#!/bin/bash

# Sequential Training Script for JAAD and PIE datasets
# This script runs the training configurations one after another

# Configuration
PYTHON_PATH="/home/minshi/miniconda3/envs/tf26/bin/python"
SCRIPT_PATH="train_test.py"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to run training
run_training() {
    local config_name=$1
    local config_path=$2
    local description=$3
    
    print_status $BLUE "=============================================================="
    print_status $GREEN "🚀 Starting: $description"
    print_status $YELLOW "Config: $config_path"
    print_status $YELLOW "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    print_status $BLUE "=============================================================="
    
    # Run the training command
    $PYTHON_PATH $SCRIPT_PATH -c $config_path
    
    # Check if the command succeeded
    if [ $? -eq 0 ]; then
        print_status $GREEN "✅ Successfully completed: $description"
        return 0
    else
        print_status $RED "❌ Failed: $description"
        return 1
    fi
}

# Main execution
main() {
    print_status $BLUE "🎯 Sequential Training Script"
    print_status $YELLOW "📁 Working directory: $(pwd)"
    print_status $YELLOW "🐍 Python environment: $PYTHON_PATH"
    
    # Check if Python environment exists
    if [ ! -f "$PYTHON_PATH" ]; then
        print_status $RED "❌ Python environment not found: $PYTHON_PATH"
        exit 1
    fi
    
    # Check if training script exists
    if [ ! -f "$SCRIPT_PATH" ]; then
        print_status $RED "❌ Training script not found: $SCRIPT_PATH"
        exit 1
    fi
    
    start_time=$(date +%s)
    
    # Training configurations
    configs=(
        "JAAD|config_files/my/my_jaad.yaml|Training on JAAD dataset with depth information"
        "PIE|config_files/my/my_pie.yaml|Training on PIE dataset with depth information"
    )
    
    successful=0
    failed=0
    
    # Run each configuration
    for i in "${!configs[@]}"; do
        IFS='|' read -r name config_path description <<< "${configs[$i]}"
        
        # Check if config file exists
        if [ ! -f "$config_path" ]; then
            print_status $RED "❌ Config file not found: $config_path"
            ((failed++))
            continue
        fi
        
        print_status $YELLOW "\n🔄 Running configuration $((i+1))/${#configs[@]}: $name"
        
        if run_training "$name" "$config_path" "$description"; then
            ((successful++))
            if [ $((i+1)) -lt ${#configs[@]} ]; then
                print_status $YELLOW "⏳ Waiting 5 seconds before next training..."
                sleep 5
            fi
        else
            ((failed++))
            read -p "Training failed. Continue with next configuration? (y/n): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                print_status $YELLOW "Training sequence aborted by user."
                break
            fi
        fi
    done
    
    # Print summary
    end_time=$(date +%s)
    total_time=$((end_time - start_time))
    hours=$((total_time / 3600))
    minutes=$(((total_time % 3600) / 60))
    
    print_status $BLUE "\n=============================================================="
    print_status $GREEN "📋 TRAINING SEQUENCE SUMMARY"
    print_status $BLUE "=============================================================="
    print_status $YELLOW "⏱️  Total time: ${hours}h ${minutes}m (${total_time}s)"
    print_status $YELLOW "📊 Configurations processed: $((successful + failed))"
    print_status $GREEN "✅ Successful: $successful"
    print_status $RED "❌ Failed: $failed"
    
    if [ $failed -gt 0 ]; then
        print_status $RED "\n⚠️ $failed configuration(s) failed."
        exit 1
    else
        print_status $GREEN "\n🎉 All trainings completed successfully!"
    fi
}

# Handle Ctrl+C
trap 'print_status $YELLOW "\n⚠️ Script interrupted by user"; exit 1' INT

# Run main function
main "$@"
