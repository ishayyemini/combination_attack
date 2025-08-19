#!/bin/sh

#SBATCH --job-name=ishay
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 2      # cores requested
#SBATCH -o outfile  # send stdout to outfile
#SBATCH -e errfile  # send stderr to errfile
#SBATCH -t 24:00:00  # time requested in hour:minute:second
#SBATCH --gpus=1
#SBATCH --partition=gpu-sharifm

python big_corpus_test.py
