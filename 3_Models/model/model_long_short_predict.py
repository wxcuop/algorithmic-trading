#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import traceback
import math
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.layers import Input, Dropout, Dense
from keras.models import Sequential
from scikeras.wrappers import KerasClassifier, KerasRegressor

yLen=2 # Target variable is one-hot encoded (i.e [0,1], [1,0] etc). it represents log/short decision
b=0

# Tensorflow verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# These are the paths to where SageMaker mounts interesting things in your
# container.
prefix = '/opt/ml/'

input_path = prefix + 'input/data/training/data_train.csv'
test_path = prefix + 'input/data/training/data_test.csv'

output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')

# Process and prepare the data
def data_process(df):
    global yLen # number of output classes
    global b  # number of input features
    dataX=[]
    dataY=[]
    for idx,row in df.iterrows():
        row1=[]
        r=row[1:len(row)-yLen]  # Exctract features
        for a in r:
            row1.append(a)
        x=np.array(row1)
        y=np.array(row[len(row)-yLen:])  # Extract targets
        b=len(x)
        dataX.append(x) # Add the feature array to the dataX list
        dataY.append(y) # Add the target array to the dataY list
    # Converts the list of feature arrays into a single NumPy (common for neural networks)        
    dataX=np.array(dataX).astype(np.float32)
    dataY=np.array(dataY).astype(np.float32)
    return dataX,dataY,b

def build_classifier():
    global b
    global yLen
    print("build_classifier:b=%s,yLen=%s" % (b,yLen))
    model = Sequential()

    # Input Layer: Defines the input shape for the model.
    model.add(Input(shape=(b,))) # For a single sample, the shape is just (b,)

    # Hidden Layer 1: Dense (fully connected) layer with 'b' neurons.
    model.add(Dense(b, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2)) # Drop 20% of the neurons from the previous layer to prevent overfitting

    # Hidden Layer 2: Dense layer with 'b/2' (half the input features) neurons.
    model.add(Dense(int(b/2), kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))

    # Output Layer: Dense layer with 'yLen' neurons (2 in this case, for binary classification).
    model.add(Dense(yLen,kernel_initializer='normal', activation='sigmoid')) # squashes cells into numbers between 0 and 1

    # Compile the model: Configure the learning process.
    # 'loss='binary_crossentropy'': Suitable for binary classification or multi-label classification where each output is independent.
    # 'optimizer='adam'': An adaptive learning rate optimization algorithm.
    # 'metrics=['accuracy']': The metric to monitor during training and evaluation.    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def generate_model(dataX, dataY, b):
    model=build_classifier()
    # Train the model using the provided training data (dataX, dataY)
    model.fit(dataX, dataY, epochs=100, batch_size=1)
    # Evaluate the model's performance on the training data.  
    scores = model.evaluate(dataX, dataY, verbose=0)
    print("Training Data %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    return model
        
def train():
    print('Starting the training.')
    try:
        # Process training data
        raw_data = pd.read_csv(input_path)
        X, y, b = data_process(raw_data)
        
        # Generate (train) the model       
        model = generate_model(X, y, b)

        # Save the trained model to the specified model path in H5 format       
        model.save(os.path.join(model_path, 'model.h5'))
        print('Training is complete. Model saved.')

        # Process test data        
        raw_data = pd.read_csv(test_path)
        testX, testY, b = data_process(raw_data)

        # How well did we predict?
        scores = model.evaluate(testX, testY, verbose=0)
        print("Test Data %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        
    except Exception as e:
        # Write out an error file. This will be returned as the failure
        # Reason in the DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(output_path, 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs
        print(
            'Exception during training: ' + str(e) + '\n' + trc,
            file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)

if __name__ == '__main__':
    train()

    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)
