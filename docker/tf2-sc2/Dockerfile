
#Loading a preexisting image, try to get as close as you can before you start building.
FROM tensorflow/tensorflow:2.3.1-gpu

#Running an update and upgrade. This is usually required before installing anything new.
RUN apt-get update && \
    apt-get install -y \
        git \
        unzip wget

RUN pip install pandas scikit-learn scipy seaborn
RUN pip install gym matplotlib

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
ENV SC2PATH /opt/StarCraftII/

WORKDIR /opt
RUN wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip && \
  unzip -P iagreetotheeula SC2.4.10.zip && \
  rm *.zip

RUN pip install pysc2
RUN pip install pip install git+https://github.com/oxwhirl/smac.git

RUN wget https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip && \
  unzip SMAC_Maps.zip -d /opt/StarCraftII/Maps && \
  rm *.zip

#Making sure that SC2 is available to all users.
RUN chmod -R 755 /opt/StarCraftII

RUN pip install gitpython
RUN pip install gym gym_minigrid
RUN pip install git+https://github.com/raide-project/ctf_public.git@wip/gazebo2
RUN pip install tensorflow_probability
