#!/bin/bash

# to check whether the docker is set up correctly, run `docker run hello-world`
# you might need to add your user to the docker group
# reboot if any permission error occurs
# c.f. https://stackoverflow.com/questions/48957195/how-to-fix-docker-permission-denied

source ~/miniconda3/etc/profile.d/conda.sh
conda activate safe
cd ~/Repositories/scg_tsung

docker build --no-cache -t safe-control-gym .
docker tag safe-control-gym:latest michlche/safe-control-gym:3D
docker push michlche/safe-control-gym:3D
