import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

def tf_clipped_log(val, vmin=1e-10, vmax=10.0):
    return tf.math.log(tf.clip_by_value(val, vmin, vmax))
