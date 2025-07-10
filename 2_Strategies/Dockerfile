FROM tensorflow/tensorflow:nightly-jupyter

LABEL description="Algorithmic trading model image"


ARG TA_LIB_VERSION=0.6.4
ARG TA_LIB_DIR=ta-lib-${TA_LIB_VERSION}

RUN apt-get -y update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    python3 \
    nginx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download, build, and install the TA-Lib C library in a single layer, then clean up
RUN wget https://github.com/ta-lib/ta-lib/releases/download/v${TA_LIB_VERSION}/${TA_LIB_DIR}-src.tar.gz && \
    tar -xzf ${TA_LIB_DIR}-src.tar.gz && \
    cd ${TA_LIB_DIR}/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd ../ && \
    rm -rf ta-lib ${TA_LIB_DIR}-src.tar.gz && \
    ldconfig


RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    numpy==2.3.1 \
    ta-lib==0.6.4 \
    scipy==1.16.0 \
    scikit-learn==1.7.0 \
    scikeras==0.13.0 \
    pandas==2.3.1 \
    flask==3.1.1 \
    gevent \
    gunicorn \
    matplotlib \
    backtrader==1.9.78.123 \
    boto3 \
    --ignore-installed blinker

# Env Variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY model /opt/program
WORKDIR /opt/program