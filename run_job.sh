#!/bin/bash

#PBS -N pangeo_test
#PBS -P dg97
#PBS -q gpursaa
#PBS -l walltime=48:00:00
#PBS -l ncpus=14
#PBS -l ngpus=1
#PBS -l mem=128GB
#PBS -l jobfs=20GB
#PBS -l storage=gdata/y89

source activate 
conda activate astero

cd /g/data/y89/jp6476/
python3 /g/data/y89/jp6476/Astroconformer/main.py \