"""
module for training
and exporting ZIPNet
with underlying pretrained VGG-16
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# applications/python3.8 install certificates.command
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from make_dataset import make_xy
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.layers.experimental.preprocessing import (
    RandomContrast,
    RandomFlip,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean, MeanAbsoluteError, RootMeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-paper")

# loading full dataset (y, X)
y, X = make_xy()

# splitting dataset into train-validation-test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1 / (1 - 0.1)
)

# iterable test dataset for memory efficiency
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# ZIPNet with pretrained VGG-16
class ZIPNet(Model):
    def __init__(self):
        super().__init__()
        tf.keras.backend.clear_session()
        # mean loss for each mini-batch
        self.loss_ = Mean(name="loss")
        # metric for evaluation
        self.mae_ = MeanAbsoluteError(name="mae")
        self.rmse_ = RootMeanSquaredError(name="rmse")
        # layers
        self.random_contrast = RandomContrast((0.1, 2))
        self.random_flip = RandomFlip("horizontal")
        self.base = tf.keras.applications.VGG16(
            include_top=False, weights="imagenet", input_shape=(224, 224, 3)
        )
        self.base.trainable = True
        fine_tune_at = 19
        for layer in self.base.layers[:fine_tune_at]:
            layer.trainable = False

        self.gap = GlobalAveragePooling2D()
        self.h1 = Dense(64, activation="relu", kernel_initializer="he_normal")
        self.h_lam_1 = Dense(16, activation="relu", kernel_initializer="he_normal")
        self.h_lam_2 = Dense(8, activation="relu", kernel_initializer="he_normal")
        self.h_pi_1 = Dense(16, activation="relu", kernel_initializer="he_normal")
        self.h_pi_2 = Dense(8, activation="relu", kernel_initializer="he_normal")
        self.lam = Dense(1, activation="exponential", kernel_initializer="he_normal")
        self.pi = Dense(1, activation="sigmoid", kernel_initializer="he_normal")

    # model
    def call(self, x):
        random_contrast = self.random_contrast(x)
        random_flip = self.random_flip(random_contrast)
        base = self.base(random_flip)
        gap = self.gap(base)
        h1 = self.h1(gap)
        h_lam_1 = self.h_lam_1(h1)
        h_lam_2 = self.h_lam_2(h_lam_1)
        h_pi_1 = self.h_pi_1(h1)
        h_pi_2 = self.h_pi_2(h_pi_1)
        lam = self.lam(h_lam_2)
        pi = self.pi(h_pi_2)
        return lam, pi

    # zip loss
    def zip_loss(self, y, lam, pi):
        lam, pi = tf.squeeze(lam), tf.squeeze(pi)
        loglik = tf.where(
            tf.math.equal(y, 0),
            tf.math.log(tf.add(pi, tf.multiply(1 - pi, tf.math.exp(-lam)))),
            tf.math.log(1 - pi)
            + tf.math.multiply(y, tf.math.log(lam))
            - lam
            - tf.math.lgamma(y + 1),
        )
        nll = -tf.reduce_mean(loglik)
        return nll

    # training step
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            lam, pi = self(x, training=True)
            loss = self.zip_loss(y, lam, pi)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_.update_state(loss)
        self.mae_.update_state(y, tf.multiply(1 - pi, lam))
        self.rmse_.update_state(y, tf.multiply(1 - pi, lam))
        return {m.name: m.result() for m in self.metrics}

    # test step for validation
    def test_step(self, data):
        x, y = data
        lam, pi = self(x, training=False)
        loss = self.zip_loss(y, lam, pi)
        self.loss_.update_state(loss)
        self.mae_.update_state(y, tf.multiply(1 - pi, lam))
        self.rmse_.update_state(y, tf.multiply(1 - pi, lam))
        return {m.name: m.result() for m in self.metrics}

    # metrics for monitoring
    @property
    def metrics(self):
        return [self.loss_, self.mae_, self.rmse_]

    # model assessment
    def assess(self, test_dataset, y_test, batch_size):
        # prediction
        y_pred = np.array([])
        for batch_idx, (X_, y_) in enumerate(test_dataset.batch(batch_size)):
            lam_, p_ = self(X_, training=False)
            y_pred_ = (1 - p_.numpy().flatten()) * lam_.numpy().flatten()
            y_pred = np.concatenate([y_pred, y_pred_])
        # evaluation criteria 1 : prediction of extreme values
        idx95 = np.array(
            np.where(
                (y_test > np.quantile(y_test, 0.95))
                | (y_test < np.quantile(y_test, 0.05))
            )
        )
        mae95 = np.mean(np.abs(y_pred[idx95] - y_test[idx95]))
        rmse95 = np.sqrt(np.mean((y_pred[idx95] - y_test[idx95]) ** 2))
        # evaluation criteria 2 : overall predictions
        mae = np.mean(np.abs(y_pred - y_test))
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
        # evaluation criteria 3 : predictive r2
        r2 = np.corrcoef(x=y_test, y=y_pred)[0, 1] ** 2
        return mae95, rmse95, mae, rmse, r2


# model training
print("training model...")
batch_size = 32
model = ZIPNet()
model.build(input_shape=(None, X_train.shape[1], X_train.shape[2], X_train.shape[3]))
model.compile(optimizer=Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999))
history = model.fit(
    x=X_train,
    y=y_train,
    epochs=1000,
    batch_size=batch_size,
    verbose=0,
    validation_data=(X_val, y_val),
    callbacks=[
        EarlyStopping(monitor="val_loss", verbose=0, patience=10),
        TensorBoard(log_dir=f"logs/", histogram_freq=1, profile_batch=10000),
        ModelCheckpoint(
            "zipnet.tf", monitor="val_loss", verbose=0, save_best_only=True
        ),
    ],
)
mae95, rmse95, mae, rmse, r2 = model.assess(test_dataset, y_test, batch_size)
test_result = {
    "model": model.name,
    "mae95": mae95,
    "rmse95": rmse95,
    "mae": mae,
    "rmse": rmse,
    "r2": r2,
}
print(f"\n test result : {test_result}")
