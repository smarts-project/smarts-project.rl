FROM ubuntu:bionic

ARG DEBIAN_FRONTEND=noninteractive

# Prevent tzdata from trying to be interactive.
ENV TZ=Europe/Minsk
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezones

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not okay.
ENV LANG C.UTF-8

# Install libraries
RUN apt-get update --fix-missing && \
    apt-get install -y \
        software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    add-apt-repository -y ppa:sumo/stable && \
    add-apt-repository -y ppa:git-core/ppa && \
    apt-get update && \
    apt-get install -y \
        git \
        libspatialindex-dev \
        python3.7 \
        python3.7-dev \
        python3.7-venv \
        sumo \
        sumo-doc \
        sumo-tools \
        wget \
        xorg && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

# Setup SUMO
ENV SUMO_HOME /usr/share/sumo

# Update default python version.
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1

# Setup virtual environment and install pip
ENV VIRTUAL_ENV=/opt/.venv
RUN python3.7 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py && \
    python get-pip.py && \
    pip install --upgrade pip

# Copy source files and install ULTRA
COPY . /src
WORKDIR /src
RUN pip install --no-cache-dir -e . && \
    cp -r /src/ultra_rl.egg-info /media/ultra_rl.egg-info

# Expose a port for Envision.
EXPOSE 8081

# Suppress message of missing /dev/input folder and copy ultra_rl.egg-info if not there
RUN echo "mkdir -p /dev/input\n" \
         "if [[ ! -d /src/ultra_rl.egg-info ]]; then" \
         "   cp -r /media/ultra_rl.egg-info /src/ultra_rl.egg-info;" \
         "   chmod -R 777 /src/ultra_rl.egg-info;" \
         "fi" >> ~/.bashrc

SHELL ["/bin/bash", "-c", "-l"]
