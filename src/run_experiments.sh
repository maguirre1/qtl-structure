#!/bin/bash
#SBATCH -J dag_qtl
#SBATCH -p pritch,normal,owners
#SBATCH --mem=8000
#SBATCH -c 1
#SBATCH -t 90:00
#SBATCH --array=0-999
#SBATCH --out logs/grn.%A_%a.out

ix=${SLURM_ARRAY_TASK_ID:=1}

python3 experiment.py ../figures/figdata/fig4_grns.large.parquet $ix $(expr $ix + 1000)
