import tensorflow as tf
from tensorflow.keras import Model
class model_2(Model):
    def __init__(self):
        super(model_2, self).__init__()
    def Model2(self):
        model2 = tf.keras.models.Sequential(
        [tf.keras.layers.Flatten(input_shape=[4,4]),
        tf.keras.layers.Dense(512, activation='relu' ),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(256, activation='relu' ),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(128, activation='relu' ),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(16, activation='relu' ),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(2, activation='softmax' )
        ])
        return model2