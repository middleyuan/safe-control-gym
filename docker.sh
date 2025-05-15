#!/bin/bash

source /home/tsung/miniconda3/etc/profile.d/conda.sh
conda activate safe
cd ~/safe-control-gym

docker build --no-cache -t safe-control-gym .
docker tag safe-control-gym:latest tsungyuan/safe-control-gym:2D
docker push tsungyuan/safe-control-gym:2D