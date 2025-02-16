#!/bin/bash

#SBATCH -A lt200349
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=4
#SBATCH -t 24:00:00

./hyperopt.sh
