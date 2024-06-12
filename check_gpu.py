import tensorflow as tf

if __name__ == "__main__":
    print(tf.__version__)
    print(tf.config.list_physical_devices("GPU"))
