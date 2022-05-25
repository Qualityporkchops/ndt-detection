# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 15:16:26 2022

@author: pierr
"""
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import keras_tuner as kt
matplotlib.rcParams.update({'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size':15
    })

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


#model-------------------------------------------------
def model_build(hp):
    hp_units = hp.Int('units', min_value=10, max_value=300, step=25)
    hp_units2 = hp.Int('units2', min_value=10, max_value=500, step=25)
    hp_units3 = hp.Int('units3', min_value=10, max_value=500, step=25)
    
    hp_dropout = hp.Choice('rate', values=[0.2,0.3,0.4,0.5])
    hp_dropout2 = hp.Choice('rate2', values=[0.2,0.3,0.4,0.5])
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=hp_units, activation = 'relu'),
        tf.keras.layers.Dense(units=hp_units2,activation = 'relu'),
        tf.keras.layers.Dropout(rate=hp_dropout),
        tf.keras.layers.Dense(units=hp_units3, activation = 'relu'),
        tf.keras.layers.Dropout(rate=hp_dropout2),
        tf.keras.layers.Dense(1)
        ])
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(
    loss = tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
    metrics = ['mse']
    )

    return model

tuner = kt.Hyperband(model_build,
                     objective='val_mse',
                     max_epochs=25,
                     factor=3,
                     overwrite = True,
                     directory="C:/Users/pierr/Documents/Programmation/",
                     project_name="hyperparams_ann"
                     )
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(train_ds, epochs=25, validation_data=val_ds, callbacks=[stop_early])
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

model = tuner.hypermodel.build(best_hps)

history = model.fit(train_ds,
          validation_data = val_ds,
          batch_size=batch,
          epochs=25,
          )
print(model.summary())
print(tuner.results_summary())

val_acc_per_epoch = history.history['val_mse']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch))+1
print('Best epoch: %d' % (best_epoch,))

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