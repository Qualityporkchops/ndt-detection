# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 15:54:04 2022

@author: Pierre
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#C:/Users/pierr/Documents/Programmation/
#C:/Users/Pierre/Documents/Programmation_icc/Python/
with np.load('C:/Users/pierr/Documents/Programmation/final_location.npz') as data:
    sig_data = data['sig_data']
    sig_location= data['sig_location']

print(sig_data.shape)
print(len(sig_location))

idx = np.arange(1000)
np.random.shuffle(idx)

sig_data = sig_data[idx]
sig_location = sig_location[idx]

#print(sig_location)

    
batch = 32
shuffle_buffer_size = 1000
seed = 42

data_ds = tf.data.Dataset.from_tensor_slices((sig_data, sig_location))
#data_ds = data_ds.shuffle(shuffle_buffer_size, reshuffle_each_iteration=None)#.batch(batch)

# =============================================================================
# train_split = 0.6
# val_split = 0.4
# test_split = 0.2
# ds_size = 1000
# =============================================================================
# =============================================================================
# train_size = int(train_split * ds_size)
# val_size = int(val_split * ds_size)
# test_size = int(test_split * ds_size)
# =============================================================================

train_ds = data_ds.take(700)
val_ds = data_ds.skip(700).take(200)
test_ds = data_ds.skip(900).take(100)

train_ds = train_ds.shuffle(shuffle_buffer_size).batch(batch)
val_ds = val_ds.shuffle(shuffle_buffer_size).batch(batch)
test_ds = test_ds.batch(batch)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation = 'relu'),
    tf.keras.layers.Dense(250,activation = 'relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(50, activation = 'relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
    ])

model.compile(
    loss = tf.keras.losses.MeanSquaredError(),
    optimizer = 'adam',
    metrics = ['mse']
    )

history = model.fit(train_ds,
          validation_data = val_ds,
          batch_size=batch,
          epochs=25,
# =============================================================================
#           callbacks=[
#               tf.keras.callbacks.EarlyStopping(
#                   monitor='val_loss',
#                   patience=2,
#                   restore_best_weights=True
#                   )
#               ]
# =============================================================================
          )

predictions = model.predict(test_ds)
print(predictions[:10])


for element in test_ds:
    print(element[1][:10])


model.evaluate(test_ds)
print(history.params)
print(history.history.keys())


#plt.plot(history.history['mse'])
plt.semilogy(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('model accuracy')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()