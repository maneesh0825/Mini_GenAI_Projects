"""
PROJECT TITLE: MNIST Digit Classification using CNN
ALGORITHM: Convolutional Neural Network
"""
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train[..., None] / 255.0
x_test = x_test[..., None] / 255.0

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(x_train, y_train, epochs=5, validation_split=0.1)

loss, accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", accuracy)
