FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
ENV HOME /root
ENV PYTHONPATH /slam_server/
ENV PYTHON_VERSION 3.8.5
ENV PYTHON_ROOT $HOME/local/python-$PYTHON_VERSION
ENV PATH $PYTHON_ROOT/bin:$PATH
ENV PYENV_ROOT $HOME/.pyenv

RUN mkdir /slam_server \
    && rm -rf /var/lib/apt/lists/*

RUN apt update && apt install -y git curl locales python3-pip python3-dev python3-passlib python3-jwt \
    libssl-dev libffi-dev zlib1g-dev libpq-dev

RUN echo "ja_JP UTF-8" > /etc/locale.gen \
    && locale-gen

RUN git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT \
    && $PYENV_ROOT/plugins/python-build/install.sh \
    && /usr/local/bin/python-build -v $PYTHON_VERSION $PYTHON_ROOT

WORKDIR /slam_server
ADD . /slam_server/
RUN LC_ALL=ja_JP.UTF-8 \
    && pip3 install -r requirements.txt
