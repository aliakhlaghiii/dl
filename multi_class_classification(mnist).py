from tensorflow import keras
from keras.datasets.mnist import load_data
from matplotlib import pyplot as plt

(x_train, y_train), (x_test, y_test) = load_data()
print(x_train.shape, x_train.dtype)    # (60000, 28, 28)
print(y_train.shape, y_train.dtype)    # (60000, )

plt.imshow(x_train[2], cmap='binary') # gray: inverted binary
plt.show()

print(x_test.shape)

# normalization
x_train = x_train/255.0
x_test = x_test/255.0

# one-hot
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# reshape
x_train = x_train.reshape((60000,28*28))
x_test = x_test.reshape((-1,28*28))

# model
model = keras.Sequential([
    keras.layers.InputLayer(shape=(28*28,)),
    keras.layers.Dense(units=64, activation=keras.activations.relu),
    keras.layers.Dense(units=32, activation=keras.activations.relu),
    keras.layers.Dense(units=16, activation=keras.activations.relu),
    keras.layers.Dense(units=10, activation=keras.activations.softmax),
])

model.summary()

# 65*32 (bias + 64 * 32 so the params are 2080)
# fit
model.compile(
    loss = keras.losses.CategoricalCrossentropy(),
    optimizer = keras.optimizers.SGD(),   # or Adam
    metrics = [keras.metrics.CategoricalAccuracy()]

)
history = model.fit(
x_train, y_train,
epochs=10,
batch_size= 32 ,
validation_data= (x_test, y_test))

# predict
y_pred = model.predict(x_test)

print(history.history.keys())

# train loss vs val loss
plt.plot(history.history['loss'], '-r')
plt.plot(history.history['val_loss'], '-g')

