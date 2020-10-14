# -*- coding: utf-8 -*-

import tensorflow as tf
import json

"""Load and prepare the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). Convert the samples from integers to floating-point numbers:"""

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

"""Build the `tf.keras.Sequential` model by stacking layers. Choose an optimizer and loss function for training:"""

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

metrics = model.evaluate(x_test,  y_test, verbose=2)

with open("eval-metrics.json", 'w') as f:
  json.dump({"loss": metrics[0], "accuracy": metrics[1]}, f)

