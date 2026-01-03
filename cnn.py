import numpy as np
import tensorflow as tf
# from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import random
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# define imports to avoid diagnostic issues
to_categorical = tf.keras.utils.to_categorical
Sequential = tf.keras.models.Sequential
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense

zero = [1,1,1,1,0, 1,0,0,1,0, 1,0,0,1,0, 1,0,0,1,0, 1,1,1,1,0]
one = [0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0]
two = [1,1,1,1,0, 0,0,0,1,0, 1,1,1,1,0, 1,0,0,0,0, 1,1,1,1,0]
three = [1,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0, 0,0,0,1,0, 1,1,1,1,0]
four = [1,0,0,1,0, 1,0,0,1,0, 1,1,1,1,0, 0,0,0,1,0, 0,0,0,1,0]
five = [1,1,1,1,0, 1,0,0,0,0, 1,1,1,1,0, 0,0,0,1,0, 1,1,1,1,0]
six = [1,1,1,1,0, 1,0,0,0,0, 1,1,1,1,0, 1,0,0,1,0, 1,1,1,1,0]
seven = [1,1,1,1,0, 0,0,0,1,0, 0,0,1,0,0, 0,1,0,0,0, 0,1,0,0,0]
eight= [1,1,1,1,0, 1,0,0,1,0, 1,1,1,1,0, 1,0,0,1,0, 1,1,1,1,0]
nine = [1,1,1,1,0, 1,0,0,1,0, 1,1,1,1,0, 0,0,0,1,0, 0,0,0,1,0]

X = np.array([zero, one, two, three, four, five, six, seven, eight, nine])
X = X.reshape(10 ,5, 5)

y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = to_categorical(y, 10)

# plt.imshow(X[3])
plt.imshow(X[3], cmap='gray', interpolation='nearest')
plt.axis('off')
plt.title('X[3] — digit 3')
plt.show()

random.seed(1)
model = Sequential([
    Conv2D(filters=1, 
    kernel_size=(2, 2),
    strides=(1, 1), 
    padding='valid', 
    input_shape=(5, 5, 1),
    use_bias=False, 
    activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(10, activation='softmax')
])
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=1000, batch_size=1, verbose=0)

model.predict(X[[0]])

# -------------------------
print("Training U-Net on toy dataset...")
