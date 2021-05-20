#Converts the data from https://zenodo.org/record/4290212/files/input_data.zip
#into the format used in TriDy
import numpy as np

X = np.load('raw_spikes.npy')
Y = np.load('stim_stream.npy')
output = [[],[],[],[],[],[],[],[]]
T = [[] for i in range(len(Y))]

for i in X:
    T[int(i[0]/200)].append(np.array([i[0]%200,i[1]]))

for i in range(len(Y)):
    output[Y[i]].append(np.array(T[i]))

np.save('spike_trains.npy',output)
