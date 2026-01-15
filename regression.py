# linear regression
import tensorflow
from tensorflow.keras.datasets import boston_housing
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

(train_x, train_y), (test_x, test_y) = boston_housing.load_data()
print(f"train_x: {train_x}")
print(f"shape of train_x: {train_x.shape}")

# normalization(scaling) with standardscaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(train_x)
x_test_scaled = scaler.transform(test_x)

# model
model = tensorflow.keras.Sequential()
model.add(tensorflow.keras.layers.InputLayer(input_shape = (13,)))
model.add(tensorflow.keras.layers.Dense(units = 8, activation = tensorflow.keras.activations.relu))
model.add(tensorflow.keras.layers.Dense(units = 4, activation = tensorflow.keras.activations.relu))
model.add(tensorflow.keras.layers.Dense(units = 1, activation = tensorflow.keras.activations.linear))

model.compile(
loss = tensorflow.keras.losses.MeanSquaredError(),
optimizer = tensorflow.keras.optimizers.SGD(),
)
# fit
model.fit(x = x_train_scaled, y=train_y, epochs=10)    # using default batch_size=32

#predict
pred_y = model.predict(x_test_scaled)

# y_true vs y_pred plot
plt.plot([0,50],[0,50], ':k')
plt.plot(test_y, pred_y, '.')
plt.show()

# evaluate
r2 = r2_score(test_y, pred_y)
print(f"R-squared value: {r2:.3f}")