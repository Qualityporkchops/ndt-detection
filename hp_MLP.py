# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 22:47:02 2022

@author: Pierre
"""
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import keras_tuner as kt
matplotlib.rcParams.update({'font.family': 'sans-serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size':15
    })

#C:/Users/pierr/Documents/Programmation/
#C:/Users/Pierre/Documents/Programmation_icc/Python/
with np.load('C:/Users/pierr/Documents/Programmation/complex8.npz') as data:
    sig_data = data['sig_data']
    sig_id= data['sig_id']

print(sig_data.shape)
print(len(sig_id))
idx = np.arange(1000)
np.random.shuffle(idx)
sig_data = sig_data[idx]
sig_id = sig_id[idx]

batch = 32
shuffle_buffer_size = 1000


data_ds = tf.data.Dataset.from_tensor_slices((sig_data, sig_id))
train_ds = data_ds.take(800)
val_ds = data_ds.skip(800).take(200)
#test_ds = data_ds.skip(900).take(100)
train_ds = train_ds.batch(batch)
val_ds = val_ds.batch(batch)
#test_ds = test_ds.batch(batch)

# =============================================================================
# i=0
# for element in test_ds:
#   print(element[0][:5].shape)
#   i+=1
# print(i)
# =============================================================================


#model-------------------------------------------------------------------------
def model_build(hp):
    hp_units = hp.Int('units', min_value=1, max_value=150, step=25)
    hp_units2 = hp.Int('units2', min_value=1, max_value=150, step=25)
    #hp_units3 = hp.Int('units3', min_value=10, max_value=200, step=25)

    hp_dropout = hp.Choice('rate', values=[0.2,0.3,0.4,0.5])
    hp_dropout2 = hp.Choice('rate2', values=[0.2,0.3,0.4,0.5])
    #hp_dropout3 = hp.Choice('rate3', values=[0.2,0.3,0.4,0.5])

    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=hp_units, activation='relu'),
        tf.keras.layers.Dropout(rate=hp_dropout),
        tf.keras.layers.Dense(units=hp_units2, activation='relu'),
        tf.keras.layers.Dropout(rate=hp_dropout2),
# =============================================================================
#         tf.keras.layers.Dense(units=hp_units3, activation='relu'),
#         tf.keras.layers.Dropout(rate=hp_dropout3),
# =============================================================================
        tf.keras.layers.Dense(1, activation='sigmoid'),
        ])
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
      loss='binary_crossentropy',
      metrics=['accuracy',
               tf.keras.metrics.FalsePositives(),
               tf.keras.metrics.FalseNegatives(),
               tf.keras.metrics.TruePositives(),
               tf.keras.metrics.TrueNegatives(),])

    return model

tuner = kt.Hyperband(model_build,
                     objective='val_accuracy',
                     max_epochs=25,
                     factor=3,
                     overwrite = False,
                     directory="C:/Users/pierr/Documents/Programmation/",
                     project_name="hyperparams_mlpnew"
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

#predictions------------------------------------------------------------------
predictions = model.predict(val_ds)
print(predictions[:10])
for element in val_ds:
    print(element[1][:10])
    
model.evaluate(train_ds)
model.evaluate(val_ds)
print(history.params)
print(history.history.keys())


plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
#plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
#plt.savefig('mlp2accuracycomplex5.pgf', bbox_inches='tight')

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
#plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
#plt.savefig('mlp2losscomplex5.pgf', bbox_inches='tight')


#model.save('saved_model/MLP2')