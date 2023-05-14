import argparse
import logging
import os
import pickle
import random
import shutil

import numpy as np
import tensorflow.compat.v2 as tf
# import tensorflow_datasets as tfds
import tensorflow_probability as tfp
# from matplotlib import pyplot as plt
# from scipy import stats
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# TODO : clean this up
# --------------------------------------------------
BATCH_SIZE = 256
TEST_BATCH_SIZE = 1
NUM_IMPORTANCE_SAMPLES = 256  # for test set marginal likelihood estimation
MISSINGNESS_TYPE = "MNAR"  # MCAR or MNAR
MISSINGNESS_COMPLEXITY = "COMPLEX"  # SIMPLE or COMPLEX
MARGINAL_LL_MC_SAMPLES = 100
DATASET = "MNIST"  # MNIST or SVHN
LIKELIHOOD = "BERNOULLI"  # BERNOULLI or LOGISTIC_MIXTURE
VERBOSE = True
# DATASET = 'SVHN' # MNIST or SVHN
# LIKELIHOOD = 'LOGISTIC_MIXTURE' # BERNOULLI or LOGISTIC_MIXTURE
LOGISTIC_MIXTURE_COMPONENTS = 1
if DATASET == "MNIST":
    Z_DIM = 50
    img_dim = 28
else:
    Z_DIM = 200
    img_dim = 32
# --------------------------------------------------


# Define the model
class Encoder(tf.keras.layers.Layer):
    """VAE encoder."""

    def __init__(self, input_shape):
        """Creates an instance of Encoder.

        Returns:
          Encoder instance.
        """
        super(Encoder, self).__init__()

        if DATASET == "MNIST":
            encoder_layers = [(32, 3, 2), (64, 3, 2)]
        else:
            encoder_layers = [(40, 3, 2), (60, 3, 2), (60, 5, 2)]

        self.conv_layers = []
        for i, (num_filters, kernel_size, strides) in enumerate(encoder_layers):
            if i == 0:
                self.conv_layers.append(
                    tf.keras.layers.Conv2D(
                        num_filters,
                        kernel_size,
                        strides=strides,
                        activation=tf.nn.relu,
                        data_format="channels_last",
                        input_shape=input_shape,
                        padding="SAME",
                    )
                )
            else:
                self.conv_layers.append(
                    tf.keras.layers.Conv2D(
                        num_filters,
                        kernel_size,
                        strides=strides,
                        activation=tf.nn.relu,
                        data_format="channels_last",
                        padding="SAME",
                    )
                )

        self.mu_proj = tf.keras.layers.Dense(Z_DIM, activation=None)
        self.sigma_proj = tf.keras.layers.Dense(Z_DIM, activation=tf.math.softplus)

    def call(self, x):
        """Computes the forward pass through the Encoder.

        Args:
          x: `Tensor`. 4-D `Tensor` of shape [batch_size, height, width, depth]
            containing the input images.

        Returns:
          A tuple of `Tensors` of shape [batch_size, z_dim] the mean and sigma
          parameters of a Gaussian distribution.
        """
        for layer in self.conv_layers:
            x = layer(x)

        x = tf.keras.layers.Flatten()(x)

        return (self.mu_proj(x), (self.sigma_proj(x) + 1e-3))


class Decoder(tf.keras.layers.Layer):
    """VAE decoder."""

    def __init__(self):
        """Creates an instance of Decoder.

        Returns:
          Decoder instance.
        """
        super(Decoder, self).__init__()

        if DATASET == "MNIST":
            self.dense = tf.keras.layers.Dense(7 * 7 * 20, activation=tf.nn.relu)
            self.reshape_shape = [-1, 7, 7, 20]
            decoder_layers = [(40, 5, 2), (20, 5, 2)]
            fine_tune_layers = [(10, 5, 1), (10, 5, 1)]
            assert LIKELIHOOD == "BERNOULLI"
            last_layer = (1, 3, 1)
        else:
            self.dense = tf.keras.layers.Dense(4 * 4 * 60, activation=tf.nn.relu)
            self.reshape_shape = [-1, 4, 4, 60]
            decoder_layers = [(60, 3, 2), (60, 3, 2), (40, 5, 2)]
            fine_tune_layers = [(30, 5, 1), (30, 5, 1)]
            if LIKELIHOOD == "LOGISTIC_MIXTURE":
                last_layer = (9 * LOGISTIC_MIXTURE_COMPONENTS, 3, 1)
            else:
                last_layer = (3, 3, 1)

        self.decoder_layers = []
        for i, (num_filters, kernel_size, strides) in enumerate(decoder_layers):
            self.decoder_layers.append(
                tf.keras.layers.Conv2DTranspose(
                    num_filters,
                    kernel_size,
                    strides=strides,
                    activation=tf.nn.relu,
                    data_format="channels_last",
                    padding="SAME",
                )
            )

        self.fine_tune_layers = []
        for i, (num_filters, kernel_size, strides) in enumerate(fine_tune_layers):
            self.fine_tune_layers.append(
                tf.keras.layers.Conv2D(
                    num_filters,
                    kernel_size,
                    strides=strides,
                    activation=tf.nn.relu,
                    data_format="channels_last",
                    padding="SAME",
                )
            )

        self.last_layer = tf.keras.layers.Conv2D(
            last_layer[0],
            last_layer[1],
            strides=last_layer[2],
            activation=None,
            data_format="channels_last",
            padding="SAME",
        )

    def call(self, x, b=None):
        """Computes the forward pass through the Decoder.

        Args:
          x: `Tensor`. 4-D `Tensor` of shape [batch_size, height, width, depth]
            containing the input images.

        Returns:
          Tuple of three `Tensors` (mean_logit, scale_logit, pi_logit)
        """
        x = self.dense(x)
        x = tf.reshape(x, self.reshape_shape)
        for layer in self.decoder_layers:
            x = layer(x)

        if b is not None:
            x = tf.concat([x, b], axis=-1)

        for layer in self.fine_tune_layers:
            x = layer(x)

        x = self.last_layer(x)

        if LIKELIHOOD == "LOGISTIC_MIXTURE":
            mean_logit = []
            scale_logit = []
            pi_logit = []
            img_channels = 3
            k = LOGISTIC_MIXTURE_COMPONENTS
            for i in range(img_channels):
                mean_logit.append(x[:, :, :, i * k : (i + 1) * k])
                scale_logit.append(
                    x[
                        :,
                        :,
                        :,
                        (img_channels * k + i * k) : (img_channels * k + (i + 1) * k),
                    ]
                )
                pi_logit.append(
                    x[
                        :,
                        :,
                        :,
                        (2 * img_channels * k + i * k) : (
                            2 * img_channels * k + (i + 1) * k
                        ),
                    ]
                )
        else:
            mean_logit = x
            scale_logit = None
            pi_logit = None

        return mean_logit, scale_logit, pi_logit


class VAE(tf.keras.Model):
    def __init__(self, input_shape):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_shape)
        self.decoder = Decoder()

    def call(self, inputs, decoder_b=None):
        mu, sigma = self.encoder(inputs)
        q_z = tfp.distributions.Normal(mu, sigma)

        z_sample = q_z.sample()
        return self.decoder(z_sample, b=decoder_b), q_z, z_sample


# --------------------------------------------------
# Training function
def train(
        run: int,
        METHOD: str
        ):
    tf.random.set_seed(run)
    np.random.seed(run)
    random.seed(run)

    print(run, METHOD)

    # Setting up metrics & optmizer
    optimizer = tf.keras.optimizers.Adam()

    train_kl_metric = tf.keras.metrics.Mean(name="train_kl")
    train_log_prob_metric = tf.keras.metrics.Mean(name="train_log_prob")
    train_imputation_log_prob_metric = tf.keras.metrics.Mean(
        name="train_imputation_log_prob"
    )
    train_elbo_metric = tf.keras.metrics.Mean(name="train_elbo")
    train_mse_xo_metric = tf.keras.metrics.Mean(name="train_mse_xo")
    train_mse_xm_metric = tf.keras.metrics.Mean(name="train_mse_xm")

    valid_kl_metric = tf.keras.metrics.Mean(name="valid_kl")
    valid_log_prob_metric = tf.keras.metrics.Mean(name="valid_log_prob")
    valid_imputation_log_prob_metric = tf.keras.metrics.Mean(
        name="valid_imputation_log_prob"
    )
    valid_elbo_metric = tf.keras.metrics.Mean(name="valid_elbo")
    valid_mse_xo_metric = tf.keras.metrics.Mean(name="valid_mse_xo")
    valid_mse_xm_metric = tf.keras.metrics.Mean(name="valid_mse_xm")

    test_kl_metric = tf.keras.metrics.Mean(name="test_kl")
    test_log_prob_metric = tf.keras.metrics.Mean(name="test_log_prob")
    test_imputation_log_prob_metric = tf.keras.metrics.Mean(
        name="test_imputation_log_prob"
    )
    test_elbo_metric = tf.keras.metrics.Mean(name="test_elbo")
    test_mse_xo_metric = tf.keras.metrics.Mean(name="test_mse_xo")
    test_mse_xm_metric = tf.keras.metrics.Mean(name="test_mse_xm")
    test_marginal_ll_metric = tf.keras.metrics.Mean(name="test_marginal_ll")
    test_bits_per_pixel_metric = tf.keras.metrics.Mean(name="test_bits_per_pixel")

    @tf.function
    def train_step(x, b, inputs, decoder_b, model):
        """Defines a single training step: Update weights based on one batch."""
        with tf.GradientTape() as tape:
            (x_logits, scale_logit, pi_logit), q_z, _ = model(inputs, decoder_b)
            loss_value, x_pred, log_prob, imputation_log_prob, kl = loss_fn(
                q_z, x, b, x_logits, scale_logit, pi_logit
            )

        grads = tape.gradient(loss_value, model.trainable_weights)
        grads, _ = tf.clip_by_global_norm(grads, 2.5)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        mse_xo, mse_xm = compute_mse(b, x, x_pred)

        train_kl_metric(kl)
        train_log_prob_metric(log_prob)
        train_imputation_log_prob_metric(imputation_log_prob)
        train_elbo_metric(-loss_value)
        train_mse_xo_metric(mse_xo)
        train_mse_xm_metric(mse_xm)

    @tf.function
    def eval_step(x, b, inputs, decoder_b, model, validation_set=False):
        """Get model predictions for one batch and update metrics."""
        (x_logits, scale_logit, pi_logit), q_z, z_sample = model(inputs, decoder_b)
        loss_value, x_pred, log_prob, imputation_log_prob, kl = loss_fn(
            q_z, x, b, x_logits, scale_logit, pi_logit
        )

        mse_xo, mse_xm = compute_mse(b, x, x_pred)

        if validation_set:
            valid_kl_metric(kl)
            valid_log_prob_metric(log_prob)
            valid_imputation_log_prob_metric(imputation_log_prob)
            valid_elbo_metric(-loss_value)
            valid_mse_xo_metric(mse_xo)
            valid_mse_xm_metric(mse_xm)
        else:
            test_kl_metric(kl)
            test_log_prob_metric(log_prob)
            test_imputation_log_prob_metric(imputation_log_prob)
            test_elbo_metric(-loss_value)
            test_mse_xo_metric(mse_xo)
            test_mse_xm_metric(mse_xm)

            marginal_ll, bits_per_pixel = compute_marginal_likelihood_estimate(
                z_sample, q_z, x, b, x_logits, scale_logit, pi_logit
            )
            test_marginal_ll_metric(marginal_ll)
            test_bits_per_pixel_metric(bits_per_pixel)

    if DATASET == "MNIST":
        if "Indicators" in METHOD:
            input_shape = [28, 28, 2]
        else:
            input_shape = [28, 28, 1]
    else:
        if "Indicators" in METHOD:
            input_shape = [32, 32, 4]
        else:
            input_shape = [32, 32, 3]

    model = VAE(input_shape)

    def get_inputs(example):
        if METHOD == "Zero Imputation":
            return example["x_zero_imp"], None
        elif METHOD == "Mean Imputation":
            return example["x_mean_imp"], None
        elif METHOD == "Zero Imputation Encoder Indicators":
            b = tf.expand_dims(example["b"][:, :, :, 0], -1)
            return tf.concat([example["x_zero_imp"], b], -1), None
        elif METHOD == "Zero Imputation Encoder Decoder Indicators":
            b = tf.expand_dims(example["b"][:, :, :, 0], -1)
            return tf.concat([example["x_zero_imp"], b], -1), b

    best_valid_elbo = None
    num_epochs_since_improvement = 0
    saved_model_loc = "best_model.h5"
    if os.path.exists(saved_model_loc):
        os.remove(saved_model_loc)

    for epoch in range(200):
        for example in train_ds:
            inputs, decoder_b = get_inputs(example)
            train_step(example["x"], example["b"], inputs, decoder_b, model)

        kl = train_kl_metric.result().numpy()
        log_prob = train_log_prob_metric.result().numpy()
        imputation_log_prob = train_imputation_log_prob_metric.result().numpy()
        elbo = train_elbo_metric.result().numpy()
        mse_xo = train_mse_xo_metric.result().numpy()
        mse_xm = train_mse_xm_metric.result().numpy()

        if VERBOSE:
            print(
                f"Train set evaluation epoch: {epoch} - "
                f"ELBO: {elbo}, KL: {kl}, log prob: {log_prob}, "
                f"imp log prob: {imputation_log_prob}, "
                f"mse_xo {mse_xo}, mse_xm: {mse_xm}"
            )

        train_kl_metric.reset_states()
        train_log_prob_metric.reset_states()
        train_imputation_log_prob_metric.reset_states()
        train_elbo_metric.reset_states()
        train_mse_xo_metric.reset_states()
        train_mse_xm_metric.reset_states()

        for example in valid_ds:
            inputs, decoder_b = get_inputs(example)
            eval_step(
                example["x"],
                example["b"],
                inputs,
                decoder_b,
                model,
                validation_set=True,
            )

        kl = valid_kl_metric.result().numpy()
        log_prob = valid_log_prob_metric.result().numpy()
        imputation_log_prob = valid_imputation_log_prob_metric.result().numpy()
        elbo = valid_elbo_metric.result().numpy()
        mse_xo = valid_mse_xo_metric.result().numpy()
        mse_xm = valid_mse_xm_metric.result().numpy()

        if VERBOSE:
            print(
                f"Valid set evaluation epoch: {epoch} - "
                f"ELBO: {elbo}, KL: {kl}, log prob: {log_prob}, "
                f"imp log prob: {imputation_log_prob}, "
                f"mse_xo {mse_xo}, mse_xm: {mse_xm}"
            )

        valid_kl_metric.reset_states()
        valid_log_prob_metric.reset_states()
        valid_imputation_log_prob_metric.reset_states()
        valid_elbo_metric.reset_states()
        valid_mse_xo_metric.reset_states()
        valid_mse_xm_metric.reset_states()

        if best_valid_elbo is None or elbo > best_valid_elbo:
            best_valid_elbo = elbo
            num_epochs_since_improvement = 0
            model.save_weights(saved_model_loc)
        elif num_epochs_since_improvement >= 10:
            break
        else:
            num_epochs_since_improvement += 1

    model.load_weights(saved_model_loc)
    for example in test_ds:
        for k in ["x", "x_zero_imp", "b", "x_mean_imp", "y"]:
            example[k] = tf.repeat(example[k], NUM_IMPORTANCE_SAMPLES, axis=0)
        inputs, decoder_b = get_inputs(example)
        eval_step(
            example["x"],
            example["b"],
            inputs,
            decoder_b,
            model,
            validation_set=False,
        )

    kl = test_kl_metric.result().numpy()
    log_prob = test_log_prob_metric.result().numpy()
    imputation_log_prob = test_imputation_log_prob_metric.result().numpy()
    elbo = test_elbo_metric.result().numpy()
    mse_xo = test_mse_xo_metric.result().numpy()
    mse_xm = test_mse_xm_metric.result().numpy()
    marginal_ll = test_marginal_ll_metric.result().numpy()
    bits_per_pixel = test_bits_per_pixel_metric.result().numpy()

    # generate datasets for representation learning
    def gen_rep_learning_datasets(ds):
        z, y = [], []
        for example in ds:
            inputs, decoder_b = get_inputs(example)
            z_mu, _ = model.encoder(inputs)
            z.append(z_mu)
            y.append(example["y"])
        z = tf.concat(z, axis=0).numpy()
        y = tf.concat(y, axis=0).numpy()
        return z, y

    z_train, y_train = gen_rep_learning_datasets(train_ds)
    z_test, y_test = gen_rep_learning_datasets(test_ds)

    # fit logistic classifier
    classifier = LogisticRegression(
        penalty="none", multi_class="multinomial", solver="lbfgs", max_iter=1000
    )
    classifier.fit(z_train, y_train)
    accuracy = classifier.score(z_test, y_test)

    # fit KNN classifier
    classifier = KNeighborsClassifier()
    classifier.fit(z_train, y_train)
    knn_accuracy = classifier.score(z_test, y_test)

    print(
        f"Test set evaluation epoch: {epoch} - "
        f"bits per pixel: {bits_per_pixel}, marginal ll: {marginal_ll}, "
        f"ELBO: {elbo}, KL: {kl}, log prob: {log_prob}, "
        f"imp log prob: {imputation_log_prob}, "
        f"mse_xo {mse_xo}, mse_xm: {mse_xm}, "
        f"accuracy: {accuracy}, knn accuracy: {knn_accuracy}"
    )

    metrics[METHOD]["kl"].append(kl)
    metrics[METHOD]["log_prob"].append(log_prob)
    metrics[METHOD]["imputation_log_prob"].append(imputation_log_prob)
    metrics[METHOD]["elbo"].append(elbo)
    metrics[METHOD]["mse_xo"].append(mse_xo)
    metrics[METHOD]["mse_xm"].append(mse_xm)
    metrics[METHOD]["marginal_ll"].append(marginal_ll)
    metrics[METHOD]["bits_per_pixel"].append(bits_per_pixel)
    metrics[METHOD]["accuracy"].append(accuracy)
    metrics[METHOD]["knn_accuracy"].append(knn_accuracy)


# util function
def log_sum_exp(x):
    """credit: https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py"""
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keepdims=True)
    return m + tf.math.log(tf.reduce_sum(tf.exp(x - m2), axis))


def log_probs(x, b, x_logits, scale_logit, pi_logit):
    if LIKELIHOOD == "BERNOULLI":
        x_pred = tf.nn.sigmoid(x_logits)
    elif LIKELIHOOD == "LOGISTIC_MIXTURE":
        img_channels = 3
        scale = [tf.exp(scale_logit[i]) + 1e-2 for i in range(img_channels)]
        pi = [tf.nn.softmax(pi_logit[i], axis=3) for i in range(img_channels)]

        if LOGISTIC_MIXTURE_COMPONENTS == 1:
            x_pred = tf.concat(
                [
                    tf.reduce_sum(x_logits[i], axis=3, keepdims=True)
                    for i in range(img_channels)
                ],
                axis=3,
            )
        else:
            x_pred = tf.concat(
                [
                    tf.reduce_sum(pi[i] * x_logits[i], axis=3, keepdims=True)
                    for i in range(img_channels)
                ],
                axis=3,
            )

    if LIKELIHOOD == "BERNOULLI":
        # valid for even real valued MNIST: http://ruishu.io/2018/03/19/bernoulli-vae/
        log_prob_full = -1 * tf.nn.sigmoid_cross_entropy_with_logits(
            labels=x, logits=x_logits
        )

        log_prob = tf.reduce_sum(b * log_prob_full, axis=[3, 2, 1])
        imputation_log_prob = tf.reduce_sum((1.0 - b) * log_prob_full, axis=[3, 2, 1])
    elif LIKELIHOOD == "LOGISTIC_MIXTURE":
        log_prob_full = []
        for i in range(img_channels):
            x_expanded = tf.concat(
                [
                    tf.expand_dims(x[:, :, :, i], axis=3)
                    for _ in range(LOGISTIC_MIXTURE_COMPONENTS)
                ],
                axis=3,
            )
            centered_x = x_expanded - x_logits[i]
            upper_in = (centered_x + (1.0 / 255.0)) / scale[i]
            upper_cdf = tf.nn.sigmoid(upper_in)
            lower_in = (centered_x - (1.0 / 255.0)) / scale[i]
            lower_cdf = tf.nn.sigmoid(lower_in)
            cdf_delta = upper_cdf - lower_cdf

            mid_in = centered_x / scale[i]
            log_pdf_mid = mid_in - scale_logit[i] - 2.0 * tf.nn.softplus(mid_in)
            log_cdf_plus = upper_in - tf.nn.softplus(upper_in)
            log_one_minus_cdf_min = -tf.nn.softplus(lower_in)

            log_prob_comp = tf.where(
                x_expanded < -0.999,
                log_cdf_plus,
                tf.where(
                    x_expanded > 0.999,
                    log_one_minus_cdf_min,
                    tf.where(
                        cdf_delta > 1e-5,
                        tf.math.log(tf.maximum(cdf_delta, 1e-12)),
                        log_pdf_mid - np.log(127.5),
                    ),
                ),
            )
            if LOGISTIC_MIXTURE_COMPONENTS > 1:
                log_prob_comp += tf.math.log(pi[i])

            log_prob_full.append(log_sum_exp(log_prob_comp))

        log_prob_full = tf.concat(
            [tf.expand_dims(log_prob_full[i], axis=3) for i in range(img_channels)],
            axis=3,
        )

        log_prob = tf.reduce_sum(b * log_prob_full, axis=[3, 2, 1])
        imputation_log_prob = tf.reduce_sum((1.0 - b) * log_prob_full, axis=[3, 2, 1])

    return x_pred, log_prob, imputation_log_prob


def compute_kl(q_z):
    p_z = tfp.distributions.Normal(
        loc=np.zeros(Z_DIM, dtype=np.float32), scale=np.ones(Z_DIM, dtype=np.float32)
    )
    return tf.reduce_mean(tf.reduce_sum(q_z.kl_divergence(p_z), axis=1))


def loss_fn(q_z, x, b, x_logits, scale_logit, pi_logit):
    x_pred, log_prob, imputation_log_prob = log_probs(
        x, b, x_logits, scale_logit, pi_logit
    )
    log_prob = tf.reduce_mean(log_prob)
    imputation_log_prob = tf.reduce_mean(imputation_log_prob)
    kl = compute_kl(q_z)
    return (-log_prob + kl, x_pred, log_prob, imputation_log_prob, kl)


def compute_mse(b, x, x_pred):
    sqe = (x - x_pred) ** 2
    mse_xo = tf.reduce_sum(b * sqe) / tf.reduce_sum(b)
    mse_xm = tf.reduce_sum((1.0 - b) * sqe) / tf.reduce_sum(1.0 - b)
    return mse_xo, mse_xm


def compute_marginal_likelihood_estimate(
    z_sample, q_z, x, b, x_logits, scale_logit, pi_logit
):
    # importance sampled estimate of the marginal likelihood
    _, log_p_x_given_z, _ = log_probs(x, b, x_logits, scale_logit, pi_logit)

    p_z = tfp.distributions.Normal(
        loc=np.zeros(Z_DIM, dtype=np.float32), scale=np.ones(Z_DIM, dtype=np.float32)
    )
    log_p_z = tf.reduce_sum(p_z.log_prob(z_sample), -1)
    log_q_z_given_x = tf.reduce_sum(q_z.log_prob(z_sample), -1)

    log_s = tf.math.log(tf.constant(NUM_IMPORTANCE_SAMPLES, tf.float32))

    marginal_ll = log_sum_exp(log_p_x_given_z + log_p_z - log_q_z_given_x) - log_s

    ln_2 = tf.math.log(tf.constant(2.0, tf.float32))
    num_obs_pixels = tf.reduce_sum(b[0, :, :, 0])
    bits_per_pixel = -(marginal_ll / ln_2) / num_obs_pixels

    return marginal_ll, bits_per_pixel


METHODS = [
    "Zero Imputation",
    "Zero Imputation Encoder Indicators",
    "Zero Imputation Encoder Decoder Indicators",
]
metrics = {
    METHOD: {
        "elbo": [],
        "kl": [],
        "log_prob": [],
        "imputation_log_prob": [],
        "mse_xo": [],
        "mse_xm": [],
        "marginal_ll": [],
        "bits_per_pixel": [],
        "accuracy": [],
        "knn_accuracy": [],
    }
    for METHOD in METHODS
}

# test code
if __name__ == "__main__":
    vae = VAE(input_shape=[28, 28, 1])

    X = tf.random.uniform(shape=(16, 28, 28, 1), minval=0, maxval=1)
    X_res = vae(X)
    print(X_res)
