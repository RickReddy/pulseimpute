#!/bin/bash
#SBATCH --mail-user=rithwik.r.guntaka@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --account=neurogroup_acc
#SBATCH --partition=pascal
#SBATCH --gres=gpu:2
#SBATCH --time=36:00:00
#SBATCH --mem=50G
#SBATCH --output=run_test.out

source /accre/arch/easybuild/software/BinDist/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate /home/guntakrr/.conda/envs/pulseimpute
module load GCC/11.3.0
python /home/guntakrr/og-pulseimpute/test_impmodel_display.py