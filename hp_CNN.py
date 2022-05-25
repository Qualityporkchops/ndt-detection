# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 17:15:11 2022

@author: pierr
"""
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import keras_tuner as kt
matplotlib.rcParams.update({'font.family': 'sans-serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size':15
    })
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#C:/Users/pierr/Documents/Programmation/
#C:/Users/Pierre/Documents/Programmation_icc/Python/
with np.load('C:/Users/Pierre/Documents/Programmation_icc/Python/final_signal_test.npz') as data:
    sig_data = data['sig_data']
    sig_id= data['sig_id']
    
print('training data shape:', sig_data.shape)
print('label shape:', len(sig_id))

plt.figure()
plt.plot(sig_data[0])

idx = np.arange(1000)
np.random.shuffle(idx)
sig_data = sig_data[idx]
sig_id = sig_id[idx]

batch = 32
shuffle_buffer_size = 1000


data_ds = tf.data.Dataset.from_tensor_slices((sig_data, sig_id))
train_ds = data_ds.take(700)
val_ds = data_ds.skip(700).take(200)
test_ds = data_ds.skip(900).take(100)
train_ds = train_ds.shuffle(shuffle_buffer_size).batch(batch)
val_ds = val_ds.shuffle(shuffle_buffer_size).batch(batch)
test_ds = test_ds.batch(batch)


#model------------------------------------------------------------------
def model_build(hp):
    input_shape = (1050, 1)
    hp_units = hp.Int('units', min_value=10, max_value=300, step=30)
    
    hp_conv = hp.Int('filter', min_value=32, max_value=256, step=32)
    hp_conv2 = hp.Int('filter2', min_value=32, max_value=256, step=32)
    
    hp_rate = hp.Choice('rate', values=[0.2,0.3,0.4,0.5])
    
    hp_pool = hp.Choice('pool', values=[2,3])
    hp_pool2 = hp.Choice('pool2', values=[2,3])
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape)),
        tf.keras.layers.Conv1D(hp_conv, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=hp_pool),
        tf.keras.layers.Conv1D(hp_conv2, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=hp_pool2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=hp_units, activation='relu'),
        tf.keras.layers.Dropout(rate=hp_rate),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
      loss='binary_crossentropy',
      metrics=['accuracy'])
    
    return model

tuner = kt.Hyperband(model_build,
                     objective='val_accuracy',
                     max_epochs=25,
                     factor=3,
                     overwrite = False,
                     directory="C:/Users/Pierre/Documents/Programmation_icc/Python/",
                     project_name="hyperparams3_cnn"
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

history = model.fit(
  train_ds,
  validation_data=val_ds,
  batch_size=batch,
  epochs=25,
  )
print(model.summary())
print(tuner.results_summary())


val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch))+1
print('Best epoch: %d' % (best_epoch,))

predictions = model.predict(test_ds)
print(predictions[:10])

for element in test_ds:
    print(element[1][:10])
    
model.evaluate(train_ds)
model.evaluate(val_ds)
print(history.params)
print(history.history.keys())

#%%
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
#plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')
plt.savefig('CNN2accuracycomplex10.pgf', bbox_inches='tight')

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
#plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.savefig('CNN2losscomplex10.pgf', bbox_inches='tight')

#%%
#model.save('saved_model/CNN2')
