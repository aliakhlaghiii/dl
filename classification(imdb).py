from tensorflow import keras
from keras.datasets import imdb
import numpy as np
from matplotlib import pyplot as plt

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
print(f"shape of x_train: {x_train.shape}")
print(f"type of x_train: {x_train.dtype}")

word_index = imdb.get_word_index()
reverse_word_indx = {val:key for key,val in word_index.items()}
' '.join([reverse_word_indx[i-3] for i in x_train[0] if i > 2]) # 0,1,2 are reserved, so we start from 3

# vectorization
def vectorize(sequences):
    result = np.zeros((25000,10000))
    for row, review in enumerate(sequences):
        for word in review:
            if word > 2:
                result[row, word-3] = 1.0         
    return result

x_train_vectorized = vectorize(x_train)
x_test_vectorized = vectorize(x_test)

# rehape
y_train_vector = np.reshape(y_train, (25000,1)).astype(float)
y_test_vector = np.reshape(y_test, (25000,1)).astype(float)

# model
model = keras.models.Sequential()
model.add(keras.layers.InputLayer(shape=(10000,), name='input'))
model.add(keras.layers.Dense(units=32, activation=keras.activations.relu, name='h1'))
model.add(keras.layers.Dense(units=8, activation=keras.activations.relu, name='h2'))
model.add(keras.layers.Dense(units=1, activation=keras.activations.sigmoid, name='output'))

model.summary()

# Stochastic Gradient Descent look it up
model.compile(
    loss = keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.SGD(),
    metrics = [keras.metrics.BinaryAccuracy()]
    )

# fit
history = model.fit(x_train_vectorized, y_train_vector, 
          epochs=10,
          batch_size=50,
          validation_data=(x_test_vectorized, y_test_vector)) # batch size should represent the dataset

print(history.history)

# predict
y_pred = model.predict(x_test_vectorized)
print(y_pred[:20])

# train accuracy vs validation accuracy plot
plt.plot( history.history['binary_accuracy'], '.-r', label='train')
plt.plot( history.history['val_binary_accuracy'], '.-g', label='test')
plt.legend()
plt.show()

# train loss vs validation loss plot
plt.plot( history.history['loss'], '.-r', label='train')
plt.plot( history.history['val_loss'], '.-g', label='test')
plt.legend()
plt.show()

