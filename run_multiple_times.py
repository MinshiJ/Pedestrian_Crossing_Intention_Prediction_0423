import subprocess
import sys
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Config file path')
    parser.add_argument('--times', type=int, default=3, help='Number of times to run')
    args = parser.parse_args()
    
    for i in range(args.times):
        print(f"\n{'='*50}")
        print(f"Running iteration {i+1}/{args.times}")
        print(f"{'='*50}")
        
        result = subprocess.run([
            sys.executable,
            'train_and_test_all_epoch_pipeline.py',
            '-c', args.config
        ])
        
        if result.returncode != 0:
            print(f"Run {i+1} failed with return code {result.returncode}")
            break
        else:
            print(f"Run {i+1} completed successfully")

if __name__ == "__main__":
    main()