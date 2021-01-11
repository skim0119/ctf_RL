from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import numpy as np

class LSTM_Flatten(Layer):
    """Does approximate rounding with Sawtooth wave."""
    def __init__(self,name='lstmreshape',**kwargs):
        super(LSTM_Flatten, self).__init__(**kwargs)

    def build(self,input_shape):
        pass

    def call(self, inputs):
        x = K.reshape(inputs,[K.shape(inputs)[0],-1,K.shape(inputs)[-1]])
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0],np.prod(input_shape[1:-1]),input_shape[-1])

    def get_config(self):
        config = super(LSTM_Flatten, self).get_config()
        return config
