#!/bin/bash
#SBATCH -J golden_desc
#SBATCH -c 4
#SBATCH --mem=4G
#SBATCH -t 24:00:00
#SBATCH -o /mnt/h200_raid5/zulfiyausmonova/logs/valid_golden_desc_full%j.out
#SBATCH -e /mnt/h200_raid5/zulfiyausmonova/logs/valid_golden_desc_full%j.err

set -euo pipefail

echo "Job started on $(hostname) at $(date)"

# Go to your project directory
cd /mnt/h200_raid5/zulfiyausmonova

# Activate virtual environment
source venv/bin/activate

cd semantic_drift

# Run your script
python validate_golden_descriptions.py

echo "Job finished at $(date)"