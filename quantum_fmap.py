import pennylane as qml
#import tensorflow as tf
import numpy as np
from PIL import Image

#from sklearn import datasets
#from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import cv2
import glob

dev = qml.device('default.qubit', wires=6)

@qml.qnode(dev, interface='tf')
def circuit(input):   # input is a 2 x 2 array which encodes the pixel values of the image 
    qml.RY(input[0, 0]/255, wires=0)  # 1st element 
    qml.RY(input[1, 0]/255, wires=1)
    qml.RY(input[1, 1]/255, wires=2)
    qml.RY(input[0, 1]/255, wires=3)
    
    
    # level 1 
    qml.Hadamard(wires = 3)
    qml.SWAP([3,2])
    qml.SWAP([2,1])
    qml.SWAP([1,0])

    # level 2 
    qml.ctrl(qml.Hadamard, control=3)(wires=2)
    qml.ctrl(qml.SWAP, control=3)(wires=[2,1])
    qml.ctrl(qml.SWAP, control=3)(wires=[1,0])

    # level 3 
    qml.MultiControlledX(wires = [2,3,4] )
    qml.ctrl(qml.Hadamard, control=4)(wires=1)
    qml.MultiControlledX(wires = [2,3,4] )
    # perm
    qml.MultiControlledX(wires = [2,3,4] )
    qml.ctrl(qml.SWAP, control=4)(wires=[1,0])
    qml.MultiControlledX(wires = [2,3,4] )

    # level 4 
    qml.MultiControlledX(wires = [2,3,4] )
    qml.MultiControlledX(wires = [1,4,5] )
    qml.ctrl(qml.Hadamard, control=5)(wires=0)
    qml.MultiControlledX(wires = [1,4,5] )
    qml.MultiControlledX(wires = [2,3,4] )
    
    # random circuit layers
    n_layers = 1 # no of random layers
    rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, 4))
    
    qml.RandomLayers(rand_params, wires=list(range(4)))

    
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


#imgs_all = np.load('/home/iisers/Documents/oral_cancer_project/Data/batch_0_non_oral.npy')
imgs_all = np.load('/home/iisers/Documents/oral_cancer_project/Data/batch_0_oral.npy')
all_convs = []
for i in range(len(imgs_all)):
    conv_generated = np.zeros((128, 128, 4))
    img_array = imgs_all[i]
    for j in range(0, 256, 2):
        for k in range(0, 256, 2):
            window = img_array[j:j+2, k:k+2]
            output = circuit(window)
            conv_generated[j//2, k//2, 0] = output[0]
            conv_generated[j//2, k//2, 1] = output[1]
            conv_generated[j//2, k//2, 2] = output[2]
            conv_generated[j//2, k//2, 3] = output[3]
            
    all_convs.append(conv_generated)
    
np.save("/home/iisers/Documents/oral_cancer_project/Data/conv_batch_0_oral.npy", all_convs)
