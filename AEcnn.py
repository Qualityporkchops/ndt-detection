# -*- coding: utf-8 -*-
"""
Created on Sun May  8 11:43:52 2022

@author: Pierre
"""
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
matplotlib.rcParams.update({'font.family': 'sans-serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size':18
    })

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import norm
#%%
dt = 5e-8
total_len = 1050 #number of index for total signal length
time_axis = np.arange(total_len)*dt

#C:/Users/pierr/Documents/Programmation/
#C:/Users/Pierre/Documents/Programmation_icc/Python/
with np.load('C:/Users/pierr/Documents/Programmation/reverse_final_signal.npz') as data:
    sig_data = data['sig_data']
    sig_id= data['sig_id']
    
idx = np.arange(1000)
np.random.shuffle(idx)
sig_data = sig_data[idx]
sig_id = sig_id[idx]

shuffle_buffer_size = 1000

train_data, test_data, train_labels, test_labels = train_test_split(
    sig_data, sig_id, test_size=0.2, random_state=shuffle_buffer_size
)

train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)

normal_train_data = train_data[train_labels]
normal_test_data = test_data[test_labels]

anomalous_train_data = train_data[~train_labels]
anomalous_test_data = test_data[~test_labels]


plt.figure()
plt.plot(normal_train_data[0])

plt.figure()
plt.plot(anomalous_train_data[0])
#%%
n_input = train_data.shape[1]
n_bottleneck = 75

class AnomalyDetector(tf.keras.Model):
    def __init__(self,latent_dim):
        super(AnomalyDetector, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1050,1)),
            tf.keras.layers.Conv1D(32, 3, activation='relu'),
            tf.keras.layers.Conv1D(64, 3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim)
            ])
        
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim)),
            tf.keras.layers.Dense(1046*64),
            tf.keras.layers.Reshape((1046,64)),
            tf.keras.layers.Conv1DTranspose(64, 3, activation='relu'),
            tf.keras.layers.Conv1DTranspose(32, 3, activation='relu'),
            tf.keras.layers.Conv1DTranspose(1, 1, activation='linear'),
            ])
        
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
model = AnomalyDetector(n_bottleneck)

model.compile(optimizer='adam',
             loss='mse',
             metrics=['mse'])

batch = 32
epochs = 25
history = model.fit(normal_train_data, normal_train_data,
                    epochs=epochs, 
                    batch_size=batch, 
                    validation_data=(test_data, test_data),
                    shuffle=True)

tf.keras.utils.plot_model(model, show_shapes=True)
print(model.summary())

plt.figure()
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
#plt.title('model mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')

encoded_data1 = model.encoder(normal_test_data).numpy()
decoded_data1 = model.decoder(encoded_data1).numpy()
#%%
plt.figure(figsize=[7,4])
plt.plot(time_axis*10**6, normal_test_data[67],'b')
plt.plot(time_axis*10**6, decoded_data1[67],'r')
plt.tick_params(axis='y',
                which='both',
                left=False,
                right=False,
                labelleft=False)
plt.xlabel("Time ($\mu$s)")
plt.legend(labels=["Input", "Reconstruction"])
#plt.savefig('caenormrecon.pgf', bbox_inches='tight')

encoded_data2 = model.encoder(anomalous_test_data).numpy()
decoded_data2 = model.decoder(encoded_data2).numpy()

plt.figure(figsize=[7,4])
plt.plot(time_axis*10**6, anomalous_test_data[67] ,'b')
plt.plot(time_axis*10**6, decoded_data2[67],'r')
plt.tick_params(axis='y',
                which='both',
                left=False,
                right=False,
                labelleft=False)
plt.xlabel("Time ($\mu$s)")
plt.legend(labels=["Input", "Reconstruction"])
#plt.savefig('caeanormrecon.pgf', bbox_inches='tight')

#%%
reconstructions = model.predict(normal_train_data)
reconstructions=np.squeeze(reconstructions, axis=-1)
print(reconstructions.shape)
train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)
reconstructions2 = model.predict(anomalous_test_data)
reconstructions2 = np.squeeze(reconstructions2, axis=-1)
test_loss = tf.keras.losses.mae(reconstructions2, anomalous_test_data)

plt.figure()
plt.hist(train_loss[None,:], bins=40, alpha=0.9, label="training")
plt.hist(test_loss[None, :], bins=40, alpha=0.9, label="test")
plt.xlabel("Reconstruction loss")
plt.ylabel("No of signals")
plt.legend(labels=["training", "test"])

plt.savefig('caehisto.pgf', bbox_inches='tight')


#%%

threshold = np.mean(train_loss) + 1*np.std(train_loss)
threshold = np.percentile(train_loss,95)
threshold = 0.015
print("Threshold: ", threshold)

plt.figure()
plt.hist(train_loss[np.newaxis,:], bins=100)
plt.xlabel("Train loss")
plt.ylabel("No of signals")
plt.show

plt.figure()
plt.hist(test_loss[None, :], bins=100)
plt.xlabel("Test loss")
plt.ylabel("No of signals")
plt.show

#%%
def predict(mod, data, threshold):
    reconstructions = np.squeeze(mod(data))
    loss = tf.keras.losses.mae(reconstructions, data)
    return tf.math.less(loss, threshold)

def print_stats(predictions, labels):
    print('labels are',test_labels[:10])
    print('predict is',predictions[:10])
    print("Accuracy = {}".format(accuracy_score(labels, predictions)))
    print("Precision = {}".format(precision_score(labels, predictions)))
    print("Recall = {}".format(recall_score(labels, predictions)))
    cm = confusion_matrix(labels, predictions)
    ax = sns.heatmap(cm, cmap='viridis', annot=True, fmt='g', cbar=False)
    ax.xaxis.set_ticklabels(['Defect', 'No Defect'])
    ax.yaxis.set_ticklabels(['Defect', 'No Defect'])
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels');
    #plt.savefig('aecnncm.pgf', bbox_inches='tight')
    
preds = predict(model, test_data, threshold)
print_stats(preds, test_labels)

