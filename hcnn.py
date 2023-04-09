import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchquantum as tq

from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import cv2

from skimage import io
from PIL import Image

# define controlled hadamard gate
sq2 = 1/np.sqrt(2)
def controlled_H(qdev, target,control):
      qdev.apply(tq.QubitUnitary(
      has_params=True,init_params=([[1,0,0,0],[0,sq2,0,sq2],[0,0,1,0],[0,sq2,0,-sq2]]),wires=[target,control]))
 
class QuanvolutionFilter(tq.QuantumModule):
  # the __init__ method initializes the quantum device, the general encoder,
  # a random quantum layer, and a measurement operator.
    def __init__(self):
        super().__init__()
        self.n_wires = 6  # two ancillas
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # encoding the input data
        self.encoder = tq.GeneralEncoder(
        [   {'input_idx': [0], 'func': 'ry', 'wires': [0]},
            {'input_idx': [1], 'func': 'ry', 'wires': [1]},
            {'input_idx': [2], 'func': 'ry', 'wires': [2]},
            {'input_idx': [3], 'func': 'ry', 'wires': [3]},])
        


        # random circuit layer 
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        #self.expval = tq.expval()
    
# x has the dimension of (batch_size, 28, 28) representing a batch of greyscale images
#The method first reshapes the input data into a 2D array of shape 
#(batch_size, 784) by concatenating adjacent 2x2 blocks of pixels.
# data is the new reshaped tensor 
    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0] # batch size
        size = 256 # height and width of an imagea
        x = x.view(bsz, size, size) # view all data 

        data_list = []

        for c in range(0, size, 2):
            for r in range(0, size, 2):
                data = torch.transpose(torch.cat((x[:, c, r], x[:, c, r+1], x[:, c+1, r], x[:, c+1, r+1])).view(4, bsz), 0, 1)
                if use_qiskit:
                    data = self.qiskit_processor.process_parameterized(
                        self.q_device, self.encoder, self.q_layer, self.measure, data)
                else:
                    self.encoder(self.q_device, data)
                    
                    #haar wavelet
                    # level 1 
                    self.q_device.h(wires = 3) 
                    self.q_device.swap([3,2])
                    self.q_device.swap([2,1])
                    self.q_device.swap([1,0])

                    # level 2 
                    controlled_H(self.q_device, target=2, control= 3)
                    self.q_device.cswap([3,2,1])
                    self.q_device.cswap([3,1,0])

                    # level 3
                    
                    self.q_device.ccx([2,3,4])
                    controlled_H(self.q_device, target=1, control= 4)
                    self.q_device.ccx([2,3,4])
                    #perm
                    self.q_device.ccx([2,3,4])
                    self.q_device.cswap([4,1,0])
                    self.q_device.ccx([2,3,4])

                    #level 4
                    self.q_device.ccx([2,3,4])
                    self.q_device.ccx([1,4,5])
                    controlled_H(self.q_device, target=0, control= 5)
                    self.q_device.ccx([1,4,5])
                    self.q_device.ccx([2,3,4])



                    self.q_layer(self.q_device)
                    data = self.measure(self.q_device)[:, :4]
                    
                    #for i in range(4):
                    #    measure_result = []
                    #    measure_result.append(tq.expval(self.q_device,wires=i, observables= tq.PauliZ(wires=i)))

                    #    data = torch.tensor([measure_result])

                    
                data_list.append(data.view(bsz, 4)) # only keep the first 4 qubits
        
        result = torch.cat(data_list, dim=1).float()
        
        return result


class HybridModel(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.qf = QuanvolutionFilter()
        self.linear = torch.nn.Linear(4*128*128, 2) 
    def forward(self, x, use_qiskit=False):
        with torch.no_grad():
          x = self.qf(x, use_qiskit)
        x = self.linear(x)
        return F.softmax(x, -1) 
    # F.log_softmax is the log of the softmax function, which is a common choice for the output of a classification model.
    
class HybridModel_without_qf(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(256*256, 2)
    
    def forward(self, x, use_qiskit=False):
        x = x.view(-1, 256*256)
        x = self.linear(x)
        return F.softmax(x, -1)
    

# importing the dataset 
class Oral_Can_Data(Dataset):
    '''Oral cancer dataset'''
    def __init__(self, csv_file, root_dir,transform = None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations) # returns the number of samples in the dataset
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        image = np.array([image])
        image = torch.tensor(image)
        can_type = self.annotations.iloc[index, 1] # labels:  1 for non-cancerous, 2 for cancerous
        can_type = torch.tensor([can_type])
        can_type = can_type

    
        sample = {'image': image, 'can_type': can_type}
        
        for index in range(len(self.annotations)):
            sample

        if self.transform:
            image = self.transform(image)
        
        return (sample) # returns the image and the label


# loading the dataset
dataset = Oral_Can_Data(csv_file= '/home/iisers/Documents/oral_cancer_project/labels .csv',root_dir='/home/iisers/Documents/oral_cancer_project/Combined_data_resized')

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_set, val_set,  test_set = torch.utils.data.random_split(dataset, [train_size,val_size,test_size])

dataflow = dict({'train' : train_set, 'valid': val_set , 'test' : test_set})

#dataflow = dict()

#for split in data_flow:
#    sampler = torch.utils.data.RandomSampler(data_flow[split])
#    dataflow[split] = torch.utils.data.DataLoader(
#        data_flow[split],
#        batch_size=50,
#        sampler=sampler,
#        num_workers=6,
#        pin_memory=True)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = HybridModel().to(device)
model_without_qf = HybridModel_without_qf().to(device)
n_epochs = 1
optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
criterion = torch.nn.CrossEntropyLoss()

accu_list1 = []
loss_list1 = []
accu_list2 = []
loss_list2 = []

def train(dataflow, model, device, optimizer):
    for i in range(2):
    #for feed_dict in dataflow['train']:
        #inputs = feed_dict['image'].to(device)
        #targets = feed_dict['can_type'].to(device)
        inputs = dataflow['train'][i]['image'].to(device)
        targets = dataflow['train'][i]['can_type'].to(device)
        #print(targets)
        #print(targets.shape)
        

        outputs = model(inputs)
        print(f'outputs: {outputs}')
        print(f'targets = {targets}')
        
        #print(outputs.shape)
        #print(outputs)

        #print(inputs)
        #loss = F.nll_loss(outputs, targets)
        loss = F.binary_cross_entropy(outputs[0][0].to(torch.float32), targets[0].to(torch.float32))
        #loss = criterion(outputs[0], targets.view(-1, 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"loss: {loss.item()}", end='\r')


def valid_test(dataflow, split, model, device, qiskit=False):
    target_all = []
    output_all = []
    with torch.no_grad():
        for i in range(2):
        #for feed_dict in dataflow[split]:
            #inputs = feed_dict['image'].to(device)
            #targets = feed_dict['can_type'].to(device)
            inputs = dataflow[split][i]['image'].to(device)
            targets = dataflow[split][i]['can_type'].to(device)

            outputs = model(inputs, use_qiskit=qiskit)

            target_all.append(targets.to(torch.float32))
            
            output_all.append(outputs[0][0].to(torch.float32))
            output_all = [torch.tensor(output_all)]
            
        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)
        
    _, indices = output_all.topk(1, dim=1)
   
    masks = indices.eq(target_all.expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    loss = F.binary_cross_entropy(output_all, target_all).item()

    print(f"{split} set accuracy: {accuracy}")
    print(f"{split} set loss: {loss}")

    return accuracy, loss

for epoch in range(1, n_epochs + 1):
    # train
    print(f"Epoch {epoch}:")
    train(dataflow, model, device, optimizer)
    print(optimizer.param_groups[0]['lr'])

    # valid
    accu, loss = valid_test(dataflow, 'test', model, device, )
    accu_list1.append(accu)
    loss_list1.append(loss)
    scheduler.step()
