import os
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense
from keras.models import Model


def get_model(model_arch: str):
    config = {"weights": "imagenet", "include_top": False, "input_shape": (256, 256, 3)}
    model = getattr(tf.keras.applications, model_arch)(**config)
    average = tf.keras.layers.GlobalAveragePooling2D()
    new_layer2 = Dense(2, activation="softmax", name="final_layer")
    return Model(model.input, new_layer2(average(model.output)))
