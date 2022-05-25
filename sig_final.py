# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:51:56 2022

@author: pierr
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.family': 'sans-serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size':18
    })

from random import sample

signal_frq = 5e6 #frequency of ultrasonic wave
sample_rate = 5e-8 #sampling rate
sample_frq = 1/sample_rate #sampling frequency
num_cycles = 5 #number of cycles for toneburst



dt = sample_rate
total_len = 1050 #number of index for total signal length
time_axis = np.arange(total_len)*dt


#toneburst---------------------------------------------

#tone_duration=1e-4; %in seconds
tone_duration = num_cycles/(signal_frq) #length of toneburst
tone_axis = np.arange(0,tone_duration,dt)
tone_len = len(tone_axis)

burst = np.cos(2*np.pi*signal_frq*tone_axis)*np.hanning(tone_len)


#add zeros to toneburst------------------------------------

sig = np.concatenate((burst, np.zeros(total_len-tone_len)))

#FFT-------------------------------------------------------

df = 1/(dt*total_len)
f = np.arange(0,int(total_len/2))*df
sig_fft = np.fft.fft(sig,n=total_len)[:int(total_len/2)]


#attenuation properties---------------------------------------------

D_grain = 400e-6


#NO DEFECT ----------------------------------------------
#wall1_fft = []
total1_fft = []
no_defect = []
signal_id1 = []
#location1 = []
 


for x in range(500):
    
    #echos1 = sample(range(1,6), 1)
    echos1 = [6]
    #material properties------------------------------------
    #a = sorted(np.random.choice(list(np.linspace(0.0005,1,.0001)),2))
    a = sample([i * 0.0001 for i in range(50, 200)],1)

    u = 5850 #speed in steel m/s
    
    thickness = a[0]
    travel_distance = thickness #material thickness in m
    travel_time = 2*travel_distance/u #time (s) taken to reach boundary and back to receiver
    
    
    
    #add an echo from the boundary--------------------------------
    wall1_fft = [sig_fft*np.exp(-2*np.pi*1j*f*(j+1)*travel_time)*np.exp(-(D_grain)**3/(u/signal_frq)**4*(j+1)*travel_distance) for j in range(echos1[0])] #delay and attenuation
    wall1 = sum(wall1_fft)
    total1_fft.append(sig_fft + wall1)
   
    #total signal---------------------------------------------

   
    no_defect.append(np.real(2 * np.fft.ifft(total1_fft[x],total_len)))
    signal_id1.append(0)
    #location1.append(0)
    
#no_defect = np.array(no_defect)

#------DEFECT------------------------------------------------
wall2_fft = []
echo2_fft = []
total2_fft = []
defect = []
signal_id2 = []
#location2 = []


for y in range(500):
    
    #echos2 = sample(range(1, 6), 1)
    echos2 = [6]
    #attenuation = sample([i * 0.01 for i in range(40,80)],1)
    attenuation = [0.50]
    #material properties------------------------------------ 
    b = sorted(sample([i * 0.0001 for i in range(50, 200)],2))
    #b1 = (sample([i * 0.0001 for i in range(100,200)],1))
    #b2 = sample([i * 0.0001 for i in range(40,round(b1[0]*10000)-50)],1)
    
    thickness2 = b[1]
    travel_distance2 = thickness2 #material thickness in m
    travel_time2 = 2*travel_distance2/u #time (s) taken to reach boundary and back to receiver
    
    defect_location = b[0]
    defect_distance = defect_location #depth of defect in m
    defect_time = 2*defect_distance/u #time for signal to reach defect and back
    
    
    #add an echo from the boundary--------------------------------

    wall2_fft = [sig_fft*np.exp(-2*np.pi*1j*f*(j+1)*travel_time2)*np.exp(-(D_grain)**3/(u/signal_frq)**4*(j+1)*travel_distance2) for j in range(echos2[0])] #delay and attenuation

    
    #add echo from defect----------------------------------------
    
    echo2_fft = [attenuation*sig_fft*np.exp(-2*np.pi*1j*f*(j+1)*defect_time)*np.exp(-(D_grain)**3/(u/signal_frq)**4*(j+1)*defect_distance) for j in range(echos2[0])]#delay and attenuation

    
    #total signal---------------------------------------------
    wall2 = sum(wall2_fft)
    echo3 = sum(echo2_fft)
    
    total2_fft.append(sig_fft + wall2 + echo3) 
    
    defect.append(np.real(2 * np.fft.ifft(total2_fft[y],total_len)))
    signal_id2.append(1)
    #location2.append(defect_location)
#defect = np.array(defect) 
    

#total signal data--------------------------------------------    
sig_data = np.vstack(no_defect + defect)
sig_id = np.array(signal_id1 + signal_id2)
#sig_location = np.array(location1 + location2)

#noise-------------------------------------------------------
def noisy(signal, snr_db):
    power = np.mean(signal**2)
    noisepower = power/(10**(snr_db/10))
    noisevar = noisepower#/(len(signal))
    noisestd = np.sqrt(noisevar)
    mean=0
    noise = np.random.normal(mean, noisestd, len(signal))
    noisy_sig = signal + noise
    return noisy_sig


noisy_signal = []
for element in sig_data:
    level = [5]#sample([5,10,15,20],1)
    noisy_signal.append(noisy(element, level[0]))
noisy_signal = np.array(noisy_signal)



#plots---------------------------------------------------------
for i in range(500,509):
    plt.subplot(3,3,509-i)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.plot(time_axis,np.real(noisy_signal[i]))
plt.show()


for i in range(300,325):
    plt.subplot(5,5,325-i)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.plot(time_axis,np.real(noisy_signal[i]))
plt.show()

#%%
plt.figure(figsize=(6,4))
plt.plot(time_axis*10**6,np.real(noisy_signal[789]))
plt.tick_params(axis='y',
                which='both',
                left=False,
                right=False,
                labelleft=False)
plt.xlabel("Time ($\mu$s)")
plt.savefig('complex_defect.pdf',bbox_inches='tight')

#%%

plt.figure(figsize=(6,4))
plt.plot(time_axis*10**6,np.real(noisy_signal[324]))
plt.tick_params(axis='y',
                which='both',
                left=False,
                right=False,
                labelleft=False)
plt.xlabel("Time ($\mu$s)")
#plt.savefig('complex_nodefect.pgf',bbox_inches='tight')

#save data------------------------------------------------------
#%%
#np.savez('final10.npz', sig_data=noisy_signal, sig_id=sig_id)
