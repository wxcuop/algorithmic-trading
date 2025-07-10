FROM tensorflow/tensorflow:nightly-jupyter

RUN apt-get -y update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    python3 \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Download, build, and install the TA-Lib C library in a single layer, then clean up
RUN wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz && \
    tar -xzf ta-lib-0.6.4-src.tar.gz && \
    cd ta-lib-0.6.4/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd ../ && \
    rm -rf ta-lib ta-lib-0.6.4-src.tar.gz && \
    ldconfig


RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    numpy \
    ta-lib \
    scipy \
    scikit-learn \
    scikeras \
    pandas \
    flask \
    gevent \
    gunicorn \
    backtrader \
    boto3 \
    --ignore-installed blinker

# Env Variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY model /opt/program
WORKDIR /opt/program

