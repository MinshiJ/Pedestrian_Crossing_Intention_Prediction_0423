#!/usr/bin/env python3
"""
Sequential training script for JAAD and PIE datasets
Runs training configurations one after another
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def run_command(command, description):
    """
    Run a command and handle its output
    Args:
        command: Command to run
        description: Description of what the command does
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"🚀 Starting: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    try:
        # Run the command and capture output
        result = subprocess.run(
            command,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=False,  # Let output go to terminal
            text=True,
            check=True
        )
        
        print(f"\n✅ Successfully completed: {description}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error in {description}")
        print(f"Return code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️ Interrupted: {description}")
        return False

def main():
    """Main function to run sequential training"""
    
    # Check if Python environment exists
    python_path = "/home/minshi/miniconda3/envs/tf26/bin/python"
    if not os.path.exists(python_path):
        print(f"❌ Python environment not found: {python_path}")
        sys.exit(1)
    
    # Training configurations
    configs = [
        {
            "name": "JAAD Dataset Training", 
            "config": "config_files/my/my_jaad.yaml",
            "description": "Training on JAAD dataset with depth information"
        },
        {
            "name": "PIE Dataset Training", 
            "config": "config_files/my/my_pie.yaml",
            "description": "Training on PIE dataset with depth information"
        }
    ]
    
    # Check if config files exist
    for config in configs:
        config_path = config["config"]
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            sys.exit(1)
    
    start_time = time.time()
    results = []
    
    print("🎯 Sequential Training Script")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🐍 Python environment: {python_path}")
    print(f"📊 Number of configurations: {len(configs)}")
    
    for i, config in enumerate(configs, 1):
        print(f"\n🔄 Running configuration {i}/{len(configs)}: {config['name']}")
        
        # Prepare command
        command = [
            python_path,
            "train_test.py",
            "-c",
            config["config"]
        ]
        
        # Run the training
        success = run_command(command, config["description"])
        results.append({
            "name": config["name"],
            "config": config["config"],
            "success": success,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        if not success:
            print(f"\n⚠️ Training failed for {config['name']}")
            response = input("Do you want to continue with the next configuration? (y/n): ").lower()
            if response != 'y':
                print("Training sequence aborted by user.")
                break
        else:
            print(f"\n⏳ Waiting 5 seconds before next training...")
            time.sleep(5)
    
    # Print summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n{'='*70}")
    print("📋 TRAINING SEQUENCE SUMMARY")
    print(f"{'='*70}")
    print(f"⏱️  Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    print(f"📊 Configurations processed: {len(results)}")
    
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    
    print(f"\n📝 Detailed Results:")
    for i, result in enumerate(results, 1):
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        print(f"  {i}. {result['name']}: {status}")
        print(f"     Config: {result['config']}")
        print(f"     Time: {result['timestamp']}")
    
    if failed > 0:
        print(f"\n⚠️ {failed} configuration(s) failed. Check logs above for details.")
        sys.exit(1)
    else:
        print(f"\n🎉 All trainings completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
