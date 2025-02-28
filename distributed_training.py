import tensorflow as tf
from tensorflow.keras import layers, models

# マルチGPU戦略の設定
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # シンプルなCNNモデルの定義
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# MNISTデータのロード
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# adjust the dimension of the data
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# モデルの学習（マルチGPUで分散学習）
model.fit(x_train, y_train, epochs=5, batch_size=64)
print("distributed training is done")