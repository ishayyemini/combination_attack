#!/bin/sh

#SBATCH --job-name=ishay
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 1      # cores requested
#SBATCH -o outfile  # send stdout to outfile
#SBATCH -e errfile  # send stderr to errfile
#SBATCH -t 24:00:00  # time requested in hour:minute:second
#SBATCH --gpus=1

python src/attack_like_gaslight.py

#python src/adv_dec_hparam_search.py --trials 5 --out_dir adv_dec_hparam_5_20 --query_samples 20 \
#--temp_min 0.9637 --temp_max 1.1046 --sample_min 96 --sample_max 113 \
#--pool_min 146 --pool_max 256 --beam_min 6 --beam_max 10 \
#--max_tokens_min 53 --max_tokens_max 63
