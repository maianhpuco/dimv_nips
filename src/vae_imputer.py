import tensorflow.compat.v2 as tf
import numpy as np
import time

from includes.VAE.train import train
from includes.VAE.dataset import generate_dataset

from sklearn.model_selection import train_test_split
# ROOT = os.environ.get("ROOT")
# sys.path.append(ROOT)


def vae_imputer(Xmiss, **kwargs):
    # print(Xmiss).shape
    # Xmiss = [sample for sample in Xmiss]
    t0 = time.time()
    X_train, X_valid = train_test_split(Xmiss, test_size=.2)

    # reshape into image
    X_train = X_train.reshape((-1, 28, 28, 1))
    X_valid = X_valid.reshape((-1, 28, 28, 1))

    train_ds = generate_dataset("MNIST", X_train, None, 32)
    valid_ds = generate_dataset("MNIST", X_valid, None, 32)

    # Fitting data
    model, get_inputs = train(
        run=1,
        method="Zero Imputation",
        train_ds=train_ds,
        valid_ds=valid_ds,
        ds_name="MNIST",
        z_dim=50,
        likelihood="BERNOULLI",
        mixture_components=1,
    )

    print("[+] Inference step")
    Xmiss = Xmiss.reshape((-1, 28, 28, 1))
    Xmiss = generate_dataset("MNIST", Xmiss, None, 32, infer=True)

    results = []
    for example in Xmiss:
        inputs, decoder_b = get_inputs(example)
        (x_logits, scale_logit, pi_logit), q_z, _ = model(inputs, decoder_b)

        x_pred = tf.nn.sigmoid(x_logits)
        x_pred = x_pred.numpy()
        results.append(x_pred)

    results = np.concatenate(results, axis=0)
    t0 = time.time() - t0

    return results, t0
