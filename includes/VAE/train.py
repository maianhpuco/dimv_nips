import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import random
from .model import VAE
from .dataset import generate_dataset
import os

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# --------------------------------------------------
# UTILS FUNCTION
# --------------------------------------------------
def log_sum_exp(x):
    """credit: https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py"""
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keepdims=True)
    return m + tf.math.log(tf.reduce_sum(tf.exp(x - m2), axis))


# NOTE: adding likelihood paramenter
def log_probs(x, b, x_logits, scale_logit, pi_logit, likelihood,
              mixture_components):
    if likelihood == "BERNOULLI":
        x_pred = tf.nn.sigmoid(x_logits)
    elif likelihood == "LOGISTIC_MIXTURE":
        img_channels = 3
        scale = [tf.exp(scale_logit[i]) + 1e-2 for i in range(img_channels)]
        pi = [tf.nn.softmax(pi_logit[i], axis=3) for i in range(img_channels)]

        if mixture_components == 1:
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

    if likelihood == "BERNOULLI":
        # valid for even real valued MNIST: http://ruishu.io/2018/03/19/bernoulli-vae/
        log_prob_full = -1 * tf.nn.sigmoid_cross_entropy_with_logits(
            labels=x, logits=x_logits)

        log_prob = tf.reduce_sum(b * log_prob_full, axis=[3, 2, 1])
        imputation_log_prob = tf.reduce_sum((1.0 - b) * log_prob_full,
                                            axis=[3, 2, 1])
    elif likelihood == "LOGISTIC_MIXTURE":
        log_prob_full = []
        for i in range(img_channels):
            x_expanded = tf.concat(
                [
                    tf.expand_dims(x[:, :, :, i], axis=3)
                    for _ in range(mixture_components)
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
            if mixture_components > 1:
                log_prob_comp += tf.math.log(pi[i])

            log_prob_full.append(log_sum_exp(log_prob_comp))

        log_prob_full = tf.concat(
            [
                tf.expand_dims(log_prob_full[i], axis=3)
                for i in range(img_channels)
            ],
            axis=3,
        )

        log_prob = tf.reduce_sum(b * log_prob_full, axis=[3, 2, 1])
        imputation_log_prob = tf.reduce_sum((1.0 - b) * log_prob_full,
                                            axis=[3, 2, 1])

    return x_pred, log_prob, imputation_log_prob


# NOTE: adding z_dim params
def compute_kl(q_z, z_dim):
    p_z = tfp.distributions.Normal(loc=np.zeros(z_dim, dtype=np.float32),
                                   scale=np.ones(z_dim, dtype=np.float32))
    return tf.reduce_mean(tf.reduce_sum(q_z.kl_divergence(p_z), axis=1))


def loss_fn(q_z, x, b, x_logits, scale_logit, pi_logit, likelihood,
            mixture_components, z_dim):
    x_pred, log_prob, imputation_log_prob = log_probs(x, b, x_logits,
                                                      scale_logit, pi_logit,
                                                      likelihood,
                                                      mixture_components)
    log_prob = tf.reduce_mean(log_prob)
    imputation_log_prob = tf.reduce_mean(imputation_log_prob)
    kl = compute_kl(q_z, z_dim)
    return (-log_prob + kl, x_pred, log_prob, imputation_log_prob, kl)


def compute_mse(b, x, x_pred):
    sqe = (x - x_pred)**2
    mse_xo = tf.reduce_sum(b * sqe) / tf.reduce_sum(b)
    mse_xm = tf.reduce_sum((1.0 - b) * sqe) / tf.reduce_sum(1.0 - b)
    return mse_xo, mse_xm


def compute_marginal_likelihood_estimate(z_sample, q_z, x, b, x_logits,
                                         scale_logit, pi_logit, z_dim):
    # importance sampled estimate of the marginal likelihood
    _, log_p_x_given_z, _ = log_probs(x, b, x_logits, scale_logit, pi_logit)

    p_z = tfp.distributions.Normal(loc=np.zeros(z_dim, dtype=np.float32),
                                   scale=np.ones(z_dim, dtype=np.float32))
    log_p_z = tf.reduce_sum(p_z.log_prob(z_sample), -1)
    log_q_z_given_x = tf.reduce_sum(q_z.log_prob(z_sample), -1)

    log_s = tf.math.log(tf.constant(256, tf.float32))

    marginal_ll = log_sum_exp(log_p_x_given_z + log_p_z -
                              log_q_z_given_x) - log_s

    ln_2 = tf.math.log(tf.constant(2.0, tf.float32))
    num_obs_pixels = tf.reduce_sum(b[0, :, :, 0])
    bits_per_pixel = -(marginal_ll / ln_2) / num_obs_pixels

    return marginal_ll, bits_per_pixel


# --------------------------------------------------
# TRAINING FUNCTION
# --------------------------------------------------


def train(
    run,
    method,
    train_ds,
    valid_ds,
    ds_name,
    z_dim,
    likelihood,
    mixture_components,
):
    """ """
    print("num epoch: 50")
    tf.random.set_seed(run)
    np.random.seed(run)
    random.seed(run)

    # Metrics & Optimizer
    print(run, method)
    optimizer = tf.keras.optimizers.legacy.Adam()

    train_kl_metric = tf.keras.metrics.Mean(name="train_kl")
    train_log_prob_metric = tf.keras.metrics.Mean(name="train_log_prob")
    train_imputation_log_prob_metric = tf.keras.metrics.Mean(
        name="train_imputation_log_prob")
    train_elbo_metric = tf.keras.metrics.Mean(name="train_elbo")
    train_mse_xo_metric = tf.keras.metrics.Mean(name="train_mse_xo")
    train_mse_xm_metric = tf.keras.metrics.Mean(name="train_mse_xm")

    valid_kl_metric = tf.keras.metrics.Mean(name="valid_kl")
    valid_log_prob_metric = tf.keras.metrics.Mean(name="valid_log_prob")
    valid_imputation_log_prob_metric = tf.keras.metrics.Mean(
        name="valid_imputation_log_prob")
    valid_elbo_metric = tf.keras.metrics.Mean(name="valid_elbo")
    valid_mse_xo_metric = tf.keras.metrics.Mean(name="valid_mse_xo")
    valid_mse_xm_metric = tf.keras.metrics.Mean(name="valid_mse_xm")

    @tf.function
    def train_step(x, b, inputs, decoder_b, model):
        """Defines a single training step: Update weights based on one batch."""
        with tf.GradientTape() as tape:
            (x_logits, scale_logit, pi_logit), q_z, _ = model(inputs, decoder_b)
            loss_value, x_pred, log_prob, imputation_log_prob, kl = loss_fn(
                q_z, x, b, x_logits, scale_logit, pi_logit, likelihood,
                mixture_components, z_dim)

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
        (x_logits, scale_logit,
         pi_logit), q_z, z_sample = model(inputs, decoder_b)
        loss_value, x_pred, log_prob, imputation_log_prob, kl = loss_fn(
            q_z, x, b, x_logits, scale_logit, pi_logit, likelihood,
            mixture_components, z_dim)

        mse_xo, mse_xm = compute_mse(b, x, x_pred)

        if validation_set:
            valid_kl_metric(kl)
            valid_log_prob_metric(log_prob)
            valid_imputation_log_prob_metric(imputation_log_prob)
            valid_elbo_metric(-loss_value)
            valid_mse_xo_metric(mse_xo)
            valid_mse_xm_metric(mse_xm)

    if ds_name == "MNIST":
        if "Indicators" in method:
            input_shape = [28, 28, 2]
        else:
            input_shape = [28, 28, 1]
    else:
        if "Indicators" in method:
            input_shape = [32, 32, 4]
        else:
            input_shape = [32, 32, 3]

    model = VAE(input_shape, z_dim, ds_name, likelihood, mixture_components)

    def get_inputs(example):
        if method == "Zero Imputation":
            return example["x_zero_imp"], None

        elif method == "Mean Imputation":
            return example["x_mean_imp"], None

        elif method == "Zero Imputation Encoder Indicators":
            b = tf.expand_dims(example["b"][:, :, :, 0], -1)
            return tf.concat([example["x_zero_imp"], b], -1), None

        elif method == "Zero Imputation Encoder Decoder Indicators":
            b = tf.expand_dims(example["b"][:, :, :, 0], -1)
            return tf.concat([example["x_zero_imp"], b], -1), b

    best_valid_elbo = None
    num_epochs_since_improvement = 0
    saved_model_loc = "best_model.h5"
    if os.path.exists(saved_model_loc):
        os.remove(saved_model_loc)

    for epoch in range(50):
        for example in train_ds:
            inputs, decoder_b = get_inputs(example)
            train_step(example["x"], example["b"], inputs, decoder_b, model)

        kl = train_kl_metric.result().numpy()
        log_prob = train_log_prob_metric.result().numpy()
        imputation_log_prob = train_imputation_log_prob_metric.result().numpy()
        elbo = train_elbo_metric.result().numpy()
        mse_xo = train_mse_xo_metric.result().numpy()
        mse_xm = train_mse_xm_metric.result().numpy()

        if True:
            print(f"Train set evaluation epoch: {epoch} - "
                  f"ELBO: {elbo}, KL: {kl}, log prob: {log_prob}, "
                  f"imp log prob: {imputation_log_prob}, "
                  f"mse_xo {mse_xo}, mse_xm: {mse_xm}")

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

        if True:
            print(f"Valid set evaluation epoch: {epoch} - "
                  f"ELBO: {elbo}, KL: {kl}, log prob: {log_prob}, "
                  f"imp log prob: {imputation_log_prob}, "
                  f"mse_xo {mse_xo}, mse_xm: {mse_xm}")

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

    return model, get_inputs
