# We need the CUDA base dockerfile to enable GPU rendering
# on hosts with GPUs.
FROM nvidia/cuda:8.0-devel-ubuntu16.04

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    cmake \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libopenmpi-dev \
    libosmesa6-dev \
    python3-pip \
    python3-numpy \
    python3-scipy \
    net-tools \
    unzip \
    vim \
    wget \
    xpra \
    xserver-xorg-dev \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

ENV LANG C.UTF-8

RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip
COPY ./mjkey.txt /root/.mujoco/
ENV LD_LIBRARY_PATH /root/.mujoco/mjpro150/bin:$LD_LIBRARY_PATH

ENV AWS_CONFIG_FILE /selfsupervised/config

COPY vendor/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

RUN rm /usr/bin/python
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
COPY ./vendor/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /selfsupervised
COPY ./requirements.txt /selfsupervised/
RUN pip install -r requirements.txt

# Delay moving in the entire code until the very end.
ENTRYPOINT ["/selfsupervised/vendor/Xdummy-entrypoint"]
CMD ["pytest"]
COPY . /selfsupervised
RUN pip install -e vendor/mujoco-py/ \
    && pip install -e vendor/baselines/
RUN python ss/import_dep.py
ENV PYTHONPATH $PYTHONPATH:/selfsupervised
