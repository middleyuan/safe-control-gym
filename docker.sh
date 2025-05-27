#!/bin/bash

# # example usage:
# # source conda (change the path depending on your conda installation)
# source /home/tsung/miniconda3/etc/profile.d/conda.sh
# # activate the conda environment
# conda activate safe
# # cd to the gym directory
# cd ~/safe-control-gym

# # to check whether the docker is set up correctly, run `docker run hello-world`
# # you might need to add your user to the docker group
# # reboot if any permission error occurs
# # c.f. https://stackoverflow.com/questions/48957195/how-to-fix-docker-permission-denied

# # might need to `docker login` first
# docker build --no-cache -t safe-control-gym .
# # syntax: docker tag local-image:tagname new-repo:tagname
# docker tag safe-control-gym:latest tsungyuan/safe-control-gym:2D
# # syntax: docker push new-repo:tagname
# docker push tsungyuan/safe-control-gym:2D


source ~/miniconda3/etc/profile.d/conda.sh
conda activate safe
cd ~/Repositories/scg_tsung

docker build --no-cache -t safe-control-gym .
docker tag safe-control-gym:latest michlche/safe-control-gym:3D
docker push michlche/safe-control-gym:3D