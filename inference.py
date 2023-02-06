import argparse

import cv2
import numpy as np
import tensorflow as tf
from loguru import logger

LABEL = {0: "Cat", 1: "Dog"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for binary classifier")
    parser.add_argument("i", type=str, help="path to image")
    parser.add_argument("m", type=str, help="path to model")
    args = parser.parse_args()
    model = tf.keras.models.load_model(args.m)
    img = cv2.imread(args.i)
    img = cv2.resize(img, (256, 256))
    img = tf.expand_dims(tf.convert_to_tensor(img / 255.0), axis=0, name=None)
    pred = LABEL[model(img).numpy().argmax()]
    logger.success(f"It is a {pred}!")
