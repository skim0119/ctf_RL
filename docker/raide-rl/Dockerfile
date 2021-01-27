
#Loading a preexisting image, try to get as close as you can before you start building.
FROM tensorflow/tensorflow:2.2.1-gpu-py3

#Running an update and upgrade. This is usually required before installing anything new.
RUN apt-get update && \
    apt-get install -y \
        git

#RUN apt-get update && \
#    apt-get install -y python3 python3-pip
#RUN apt-get update
#Adding custom software and packages.
#RUN pip3 install tensorflow
RUN pip install pandas scikit-learn scipy seaborn
RUN pip install gym gym-minigrid matplotlib
RUN pip install git+https://github.com/raide-project/ctf_public.git@wip/gazebo2

RUN apt-get install -y libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg curl patchelf libglfw3 libglfw3-dev cmake zlib1g zlib1g-dev swig
RUN apt-get install -y mesa-utils

RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
  && rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

#Adding entrypoint for things that need to be executed when the image is mounted
# into a container. Insert any of those arguments there.
#COPY entrypoint.sh /entrypoint.sh
#RUN chmod +x /entrypoint.sh
#ENTRYPOINT ["/entrypoint.sh"]
