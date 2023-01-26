import argparse
import itertools
import os
from collections import defaultdict
from typing import List

import pandas as pd
import tensorflow as tf
from loguru import logger
from sklearn.model_selection import StratifiedKFold

from helper import merge_cats_dogs
from models import get_model

# To disable the uage of GPU enable the comment below
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
class Train:
    def __init__(
        self,
        root_dir: str,
    ):
        self.root_dir = root_dir
        self.log = defaultdict(list)

    def _merge(self):
        logger.info("Merging the binary classes >>>>>>>>>>>>>>>>>")
        merge_cats_dogs(
            cat_dir=os.path.join(self.root_dir, "Cats"),
            dog_dir=os.path.join(self.root_dir, "Dogs"),
        )

    def fit(
        self,
        model_arc: List[str],
        epochs: List[int],
        lr: List[float],
        bs: List[int],
        folds: int = 5,
    ):
        # Create merged folder
        self._merge()
        self.merged = "dataset"
        imgen = tf.keras.preprocessing.image.ImageDataGenerator(
            # samplewise_std_normalization=True,
            horizontal_flip=True,
            rescale=1.0 / 255,
            rotation_range=10,
        )

        dd = defaultdict(list)
        # creating csv for performing stratified k-split
        for i in os.listdir(str(self.merged)):
            if i.startswith("dogs_"):
                dd["filenames"].append(i)
                dd["labels"].append("dog")
            else:
                try:
                    assert i.startswith(
                        "cats_"
                    ), "Label outside of cats and dogs detected!"
                except:
                    print()
                dd["filenames"].append(i)
                dd["labels"].append("cat")

        df = pd.DataFrame(dd)
        filename = df["filenames"]
        label = df["labels"]
        skf = StratifiedKFold(n_splits=folds)

        # All possible combinations
        combinations = list(itertools.product(model_arc, lr, bs))
        for sampled_model, sampled_lr, sampled_bs in combinations:

            # separate callbacks
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    "checkpoint"
                    + f"_{sampled_model}__"
                    + str(sampled_bs)
                    + str(sampled_lr),
                    verbose=1,
                    save_best_only=True,
                    mode="min",
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir="logs"
                    + f"_{sampled_model}__"
                    + str(sampled_bs)
                    + str(sampled_lr)
                ),
            ]
            fold_metric = []
            for i, (train_index, val_index) in enumerate(skf.split(filename, label)):
                # Initialization of model
                logger.info(
                    f" Using Model - {sampled_model}, lr - {sampled_lr}, bs - {sampled_bs}"
                )
                self.model = get_model(sampled_model)
                logger.debug(f"Running fold {i} >>>>>>>>>>>>>>>>>>>>>")
                df_train = df.iloc[train_index]
                df_val = df.iloc[val_index]
                train_dataset = imgen.flow_from_dataframe(
                    dataframe=df_train,
                    directory=self.merged,
                    x_col="filenames",
                    y_col="labels",
                    batch_size=sampled_bs,
                )
                val_dataset = imgen.flow_from_dataframe(
                    dataframe=df_val,
                    directory=self.merged,
                    x_col="filenames",
                    y_col="labels",
                    batch_size=sampled_bs,
                )
                self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=sampled_lr
                    ),  # Optimizer
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                    metrics=["categorical_accuracy"],
                    run_eagerly=False,
                )
                history = self.model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    epochs=epochs,
                    callbacks=callbacks,
                )
                fold_metric.append(history.history["val_categorical_accuracy"][0])

            mean_metric = sum(fold_metric) / folds
            logger.success(f"Training completed with the mean accuray-- {mean_metric}")

            # Logging
            self.log["model"].append(sampled_model)
            self.log["accuracy"].append(mean_metric)
            self.log["learning_rate"].append(sampled_lr)
            self.log["batch_size"].append(sampled_bs)

        pd.DataFrame(dict(self.log)).to_csv(
            "complete_log.csv",
            index=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image classification task")
    parser.add_argument(
        "--imdir",
        type=str,
        help="image dir containd Cats and Dogs as sub dirs",
        default="/home/pop/Downloads/CatsDogs"
    )

    parser.add_argument(
        "--lr",
        type=List,
        help="learning rate",
        default=[0.01, 0.0001, 0.00001],
    )

    parser.add_argument(
        "--m",
        type=List,
        help="architecture name",
        default=[
            "MobileNetV3Small",
            "EfficientNetV2B2",
            "VGG16",
            "InceptionV3",
        ],
    )

    parser.add_argument(
        "--bs",
        type=List,
        help="batch size",
        default=[18, 24],
    )

    parser.add_argument(
        "--epoch",
        type=int,
        help="No. of epochs",
        default=5,
    )
    args = parser.parse_args()
    Train(root_dir=args.imdir).fit(
        model_arc=args.m, epochs=args.epoch, lr=args.lr, bs=args.bs
    )
