import tensorflow as tf

class ConvLSTMModel(tf.keras.Model):
    def __init__(self, n_frames=4, height=256, width=256, in_channels=1, n_stacked_convlstm_layers=3):
        super(ConvLSTMModel, self).__init__()
        
        self.n_stacked_convlstm_layers = n_stacked_convlstm_layers
        
        # First (n-1) ConvLSTM layers with return_sequences=True
        self.convlstm_stack = [
            tf.keras.layers.ConvLSTM2D(
                filters=32,
                kernel_size=(3, 3),
                padding="same",
                return_sequences=True,
                activation="relu"
            ) for _ in range(n_stacked_convlstm_layers - 1)
        ]

        # Final ConvLSTM layer with return_sequences=False
        self.final_convlstm = tf.keras.layers.ConvLSTM2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=False,
            activation="relu"
        )

        # Fully connected layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation="relu")
        self.dense2 = tf.keras.layers.Dense(16, activation="relu")
        self.output_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        x = inputs
        for layer in self.convlstm_stack:
            x = layer(x)
        x = self.final_convlstm(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)

if __name__ == "__main__":
    import numpy as np
    # Below code tests out the model architecture.
    n_frames, height, width, in_channels = 4, 256, 256, 1
    model = ConvLSTMModel(n_frames, height, width, in_channels, n_stacked_convlstm_layers=3)

    # Build the model (this step is optional but useful for summary)
    model.build(input_shape=(None, n_frames, height, width, in_channels))
    x = np.random.rand(32, n_frames, height, width, in_channels)
    model(x)
    print(model.summary())