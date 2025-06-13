FROM tensorflow/tensorflow:2.19.0


RUN apt-get -y update && \
         apt-get install -y --no-install-recommends \
         build-essential \
         wget \
         python3 \
         nginx \
    && rm -rf /var/lib/apt/lists/*


RUN wget  https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz && \
          tar -xzf ta-lib-0.6.4-src.tar.gz  && cd ta-lib-0.6.4 && \
         ./configure && make && make install && cd ../ && rm -rf ta-lib-0.6.4 && rm ta-lib-0.6.4-src.tar.gz


RUN python3 -m pip install --upgrade pip
RUN pip install numpy
RUN pip install ta-lib
RUN pip install scipy
RUN pip install scikit-learn
RUN pip install pandas
RUN pip install flask
RUN pip install gevent
RUN pip install gunicorn
RUN pip install backtrader
RUN pip install boto3

# Env Variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY model /opt/program
WORKDIR /opt/program
