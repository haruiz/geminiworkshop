import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import ops
from keras.layers import Input, Dense, Layer
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # Neural Network Parameters
    batch_size, n_epoch = 100, 50
    n_hidden, latent_dim = 256, 2

    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    x_train_flat = x_train.reshape((x_train.shape[0], -1))
    x_test_flat = x_test.reshape((x_test.shape[0], -1))
    mnist_digits = (
        np.concatenate([x_train_flat, x_test_flat], axis=0).astype("float32") / 255
    )

    class Sampling(Layer):
        """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.seed_generator = keras.random.SeedGenerator(1337)

        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = ops.shape(z_mean)[0]
            dim = ops.shape(z_mean)[1]
            epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
            return z_mean + ops.exp(0.5 * z_log_var) * epsilon

    # Encoder - from 784->256->128->2
    encoder_inputs = keras.Input(shape=(28 * 28,))
    x = Dense(n_hidden, activation="relu", name="h1")(
        encoder_inputs
    )  # first hidden layer
    x = Dense(n_hidden // 2, activation="relu", name="h2")(x)  # second hidden layer

    # hidden state, which we will pass into the Model to get the Encoder.
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling(name="z")([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    # Decoder - from 2->128->256->784
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(n_hidden // 2, activation="relu")(latent_inputs)
    x = Dense(n_hidden, activation="relu")(x)
    decoder_outputs = Dense(28 * 28, activation="sigmoid")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    class VAE(keras.Model):
        def __init__(self, encoder, decoder, **kwargs):
            super().__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
            self.reconstruction_loss_tracker = keras.metrics.Mean(
                name="reconstruction_loss"
            )
            self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        @property
        def metrics(self):
            return [
                self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
            ]

        def train_step(self, data):
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = self.encoder(data)
                reconstruction = self.decoder(z)
                reconstruction_loss = ops.mean(
                    ops.sum(keras.losses.binary_crossentropy(data, reconstruction))
                )
                kl_loss = -0.5 * (
                    1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var)
                )
                kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
                total_loss = reconstruction_loss + kl_loss
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
            }

    vae = VAE(encoder, decoder)
    # vae.compile(optimizer=keras.optimizers.Adam())
    # vae.fit(mnist_digits, epochs=30, batch_size=128)
    # vae.build(mnist_digits.shape)
    # vae.save_weights("vae.weights.h5")

    vae.load_weights("vae.weights.h5")

    # Plot of the digit classes in the latent space
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_train = np.expand_dims(x_train, -1).astype("float32") / 255
    z_mean, z_log_var, z = vae.encoder.predict(x_train, verbose=0)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_train)

    df = pd.DataFrame(columns=["z[0]", "z[1]", "class"])
    df["z[0]"] = z_mean[:, 0]
    df["z[1]"] = z_mean[:, 1]
    df["class"] = y_train

    groups = df.groupby("class").head(5)
    for i, row in groups.iterrows():
        plt.text(row["z[0]"], row["z[1]"], str(row["class"]), fontsize=12, color="red")

    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()
