from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy,mean_squared_error
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import binary_accuracy

def site_acc(y_true, y_pred):
    y_pred=K.round(y_pred)
    z=K.cast(K.sum(y_true)+K.sum(y_pred)==0,dtype='float32')
    r=(1-z)*categorical_accuracy(y_true, y_pred)
    return r
