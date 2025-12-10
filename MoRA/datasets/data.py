import torch
import numpy as np
import random
import os
import json
from scipy.signal import resample
import clip
    
def select_samples(data, labels, k, name, data_path):
    unique_labels = torch.unique(labels)
    selected_data = []
    selected_labels = []
    all_indices = torch.load(f'{data_path}/few_shot_data_2/{name}_k={k}.pth')

    for i, label in enumerate(unique_labels):
        selected_indices = all_indices[i]
        selected_data.append(data[selected_indices])
        selected_labels.append(labels[selected_indices])

    selected_data = torch.cat(selected_data, dim=0)
    selected_labels = torch.cat(selected_labels, dim=0)

    return selected_data, selected_labels

def combine_load(dataset, padding_size, data_path, mode='test', k=None):

    X = np.load(f'{data_path}/{dataset}/X_{mode}.npy')
    real_labels = torch.from_numpy(np.load(f'{data_path}/{dataset}/y_{mode}.npy'))
    with open(f'{data_path}/{dataset}/{dataset}.json', 'r') as file:
        data = json.load(file)
    all_X = np.zeros((X.shape[0], X.shape[1], 22, 6))

    if dataset == 'PAMAP':
        all_X[:,:,21] = np.concatenate((X[:,:,0:3], X[:,:,3:6]), axis=-1) 
        all_X[:,:,11] = np.concatenate((X[:,:,18:21], X[:,:,21:24]), axis=-1)
        all_X[:,:,7] = np.concatenate((X[:,:,9:12], X[:,:,12:15]), axis=-1)
        original_sampling_rate = 100
        num_classes = 12

    elif dataset == 'USC-HAD':
        all_X[:,:,5] = np.concatenate((X[:,:,0:3] * 9.80665, X[:,:,3:6] / 180 * np.pi), axis=-1)
        original_sampling_rate = 100
        num_classes = 12

    elif dataset == 'UCI-HAR':
        all_X[:,:,9] = np.concatenate((X[:,:,6:9] * 9.80665, X[:,:,3:6]), axis=-1) # linear accel, gyro, total accel
        original_sampling_rate = 50
        num_classes = 6

    elif dataset == 'WISDM':
        all_X[:,:,21] = np.concatenate((X[:,:,0:3], X[:,:,3:6]), axis=-1) 
        original_sampling_rate = 20
        num_classes = 18

    elif dataset == 'DSADS':
        all_X[:,:,11] = np.concatenate((X[:,:,0:3], X[:,:,3:6]), axis=-1) 
        all_X[:,:,21] = np.concatenate((X[:,:,9:12], X[:,:,12:15]), axis=-1) 
        all_X[:,:,17] = np.concatenate((X[:,:,18:21], X[:,:,21:24]), axis=-1)
        all_X[:,:,6] = np.concatenate((X[:,:,27:30], X[:,:,30:33]), axis=-1)
        all_X[:,:,2] = np.concatenate((X[:,:,36:39], X[:,:,39:42]), axis=-1)
        original_sampling_rate = 25
        num_classes = 19

    elif dataset == 'UTD-MHAD':
        all_X[real_labels < 21,:,21,:] = np.concatenate((X[real_labels < 21,:,0:3] * 9.80665, X[real_labels < 21,:,3:6] / 180 * np.pi), axis=-1)
        all_X[real_labels >= 21,:,5,:] = np.concatenate((X[real_labels >= 21,:,0:3] * 9.80665, X[real_labels >= 21,:,3:6] / 180 * np.pi), axis=-1)
        original_sampling_rate = 50
        num_classes = 27

    elif dataset == 'MotionSense':
        all_X[:,:,5] = np.concatenate((X[:,:,:3] * 9.80665, X[:,:,3:6]), axis=-1)
        all_X[:,:,1] = np.concatenate((X[:,:,:3] * 9.80665, X[:,:,3:6]), axis=-1)
        original_sampling_rate = 50
        num_classes = 6

    elif dataset == 'Shoaib':
        all_X[:,:,1] = X[:,:,:6]
        all_X[:,:,5] = X[:,:,6:12]
        all_X[:,:,21] = X[:,:,12:18]
        all_X[:,:,20] = X[:,:,18:24]
        all_X[:,:,0] = X[:,:,24:30]
        original_sampling_rate = 50
        num_classes = 7

    elif dataset == 'MMAct':
        all_X[:,:,5] = np.concatenate((X[:,:,:3], X[:,:,3:6]), axis=-1)
        all_X[:,:,21,:3] = X[:,:,6:9]
        original_sampling_rate = 50
        num_classes = 35
    
    elif dataset == 'RealWorld':
        all_X[:,:,14] = np.concatenate((X[:,:,:3], X[:,:,3:6]), axis=-1)
        all_X[:,:,16] = np.concatenate((X[:,:,6:9], X[:,:,9:12]), axis=-1)
        all_X[:,:,13] = np.concatenate((X[:,:,12:15], X[:,:,15:18]), axis=-1)
        all_X[:,:,3] = np.concatenate((X[:,:,18:21], X[:,:,21:24]), axis=-1)
        all_X[:,:,1] = np.concatenate((X[:,:,24:27], X[:,:,27:30]), axis=-1)
        all_X[:,:,15] = np.concatenate((X[:,:,30:33], X[:,:,33:36]), axis=-1)
        all_X[:,:,9] = np.concatenate((X[:,:,36:39], X[:,:,39:42]), axis=-1)
        original_sampling_rate = 50
        num_classes = 8
    
    elif dataset == 'TNDA-HAR':
        all_X[:,:,20] = np.concatenate((X[:,:,:3], X[:,:,3:6]), axis=-1)
        all_X[:,:,2] = np.concatenate((X[:,:,6:9], X[:,:,9:12]), axis=-1)
        all_X[:,:,21] = np.concatenate((X[:,:,12:15], X[:,:,15:18]), axis=-1)
        all_X[:,:,3] = np.concatenate((X[:,:,18:21], X[:,:,21:24]), axis=-1)
        all_X[:,:,11] = np.concatenate((X[:,:,24:27], X[:,:,27:30]), axis=-1)
        original_sampling_rate = 50
        num_classes = 8
    
    all_X = all_X.reshape(all_X.shape[0], all_X.shape[1], 22 * 6)

    # resample real data to 20 Hz
    new_sampling_rate = 20
    new_length = int((all_X.shape[1] / original_sampling_rate) * new_sampling_rate)
    resampled_data = np.array([resample(sequence, new_length) for sequence in all_X])


    if resampled_data.shape[1] < padding_size:
        resampled_data = np.pad(resampled_data, ((0, 0), (0, padding_size - resampled_data.shape[1]), (0, 0)), 'wrap') # N, 200, 6
    real_inputs = torch.from_numpy(resampled_data[:,:padding_size,:]).float()  

    if mode == 'train' and k and k < len(real_inputs):
        real_inputs, real_labels = select_samples(real_inputs, real_labels, k, dataset, data_path)
    b, t, _ = real_inputs.shape
    real_inputs = real_inputs.reshape(b, t, 22, -1).permute(0, 3, 1, 2).unsqueeze(-1)
    real_inputs = real_inputs[:,:3,:,:]
    # nonzero_mask = (real_inputs != 0)
    # first_nonzero_indices = nonzero_mask.float().argmax(dim=3)
    # first_index_value = first_nonzero_indices[0, 0, 0].item()
    print(f"{dataset}-SHAPE: {real_inputs.shape}, {real_labels.shape}]")
 
    # load text
    label_dictionary = data['label_dictionary']
    label_list = [' '.join(labels) for labels in label_dictionary.values()]
    all_text = clip.tokenize(label_list).cuda()
    
    return real_inputs, real_labels, label_list, all_text, num_classes

def split_load(dataset, padding_size, data_path, mode='test', k=None):

    X = np.load(f'{data_path}/{dataset}/X_{mode}.npy')
    real_labels = torch.from_numpy(np.load(f'{data_path}/{dataset}/y_{mode}.npy'))
    with open(f'{data_path}/{dataset}/{dataset}.json', 'r') as file:
        data = json.load(file)
    # all_X = np.zeros((X.shape[0], X.shape[1], 22, 6))

    if dataset == 'PAMAP':
        channel1 = np.concatenate((X[:,:,0:3], X[:,:,3:6]), axis=-1)
        channel2 = np.concatenate((X[:,:,18:21], X[:,:,21:24]), axis=-1)
        channel3 = np.concatenate((X[:,:,9:12], X[:,:,12:15]), axis=-1)
        all_X = np.stack([channel1, channel2, channel3], axis=1)  # [N, 3, T, 6]
        original_sampling_rate = 100
        num_classes = 12

    elif dataset == 'USC-HAD':
        channel = np.concatenate((X[:,:,0:3] * 9.80665, X[:,:,3:6] / 180 * np.pi), axis=-1)
        all_X = np.expand_dims(channel, axis=1)  # [N, 1, T, 6]
        original_sampling_rate = 100
        num_classes = 12

    elif dataset == 'UCI-HAR':
        channel = np.concatenate((X[:,:,6:9] * 9.80665, X[:,:,3:6]), axis=-1)
        all_X = np.expand_dims(channel, axis=1)  # [N, 1, T, 6]
        original_sampling_rate = 50
        num_classes = 6

    elif dataset == 'WISDM':
        channel = np.concatenate((X[:,:,0:3], X[:,:,3:6]), axis=-1)
        all_X = np.expand_dims(channel, axis=1)  # [N, 1, T, 6]
        original_sampling_rate = 20
        num_classes = 18

    elif dataset == 'DSADS':
        channel1 = np.concatenate((X[:,:,0:3], X[:,:,3:6]), axis=-1)
        channel2 = np.concatenate((X[:,:,9:12], X[:,:,12:15]), axis=-1)
        channel3 = np.concatenate((X[:,:,18:21], X[:,:,21:24]), axis=-1)
        channel4 = np.concatenate((X[:,:,27:30], X[:,:,30:33]), axis=-1)
        channel5 = np.concatenate((X[:,:,36:39], X[:,:,39:42]), axis=-1)
        all_X = np.stack([channel1, channel2, channel3, channel4, channel5], axis=1)  # [N, 5, T, 6]
        original_sampling_rate = 25
        num_classes = 19

    elif dataset == 'UTD-MHAD':
        channel1 = np.zeros_like(X[:,:,:6])
        channel1[real_labels < 21] = np.concatenate((X[real_labels < 21,:,0:3] * 9.80665, X[real_labels < 21,:,3:6] / 180 * np.pi), axis=-1)
        
        channel2 = np.zeros_like(X[:,:,:6])
        channel2[real_labels >= 21] = np.concatenate((X[real_labels >= 21,:,0:3] * 9.80665, X[real_labels >= 21,:,3:6] / 180 * np.pi), axis=-1)
        
        all_X = np.stack([channel1, channel2], axis=1)  # [N, 2, T, 6]
        original_sampling_rate = 50
        num_classes = 27

    elif dataset == 'MotionSense':
        channel1 = np.concatenate((X[:,:,:3] * 9.80665, X[:,:,3:6]), axis=-1)
        channel2 = np.concatenate((X[:,:,:3] * 9.80665, X[:,:,3:6]), axis=-1)
        all_X = np.stack([channel1, channel2], axis=1)  # [N, 2, T, 6]
        original_sampling_rate = 50
        num_classes = 6

    elif dataset == 'Shoaib':
        channel1 = X[:,:,:6]
        channel2 = X[:,:,6:12]
        channel3 = X[:,:,12:18]
        channel4 = X[:,:,18:24]
        channel5 = X[:,:,24:30]
        all_X = np.stack([channel1, channel2, channel3, channel4, channel5], axis=1)  # [N, 5, T, 6]
        original_sampling_rate = 50
        num_classes = 7

    elif dataset == 'MMAct':
        channel1 = np.concatenate((X[:,:,:3], X[:,:,3:6]), axis=-1)
        channel2 = np.zeros_like(X[:,:,:6])
        channel2[:,:,:3] = X[:,:,6:9]
        all_X = np.stack([channel1, channel2], axis=1)  # [N, 2, T, 6]
        original_sampling_rate = 50
        num_classes = 35
    
    elif dataset == 'RealWorld':
        channel1 = np.concatenate((X[:,:,:3], X[:,:,3:6]), axis=-1)
        channel2 = np.concatenate((X[:,:,6:9], X[:,:,9:12]), axis=-1)
        channel3 = np.concatenate((X[:,:,12:15], X[:,:,15:18]), axis=-1)
        channel4 = np.concatenate((X[:,:,18:21], X[:,:,21:24]), axis=-1)
        channel5 = np.concatenate((X[:,:,24:27], X[:,:,27:30]), axis=-1)
        channel6 = np.concatenate((X[:,:,30:33], X[:,:,33:36]), axis=-1)
        channel7 = np.concatenate((X[:,:,36:39], X[:,:,39:42]), axis=-1)
        all_X = np.stack([channel1, channel2, channel3, channel4, channel5, channel6, channel7], axis=1)  # [N, 7, T, 6]
        original_sampling_rate = 50
        num_classes = 8
    
    elif dataset == 'TNDA-HAR':
        channel1 = np.concatenate((X[:,:,:3], X[:,:,3:6]), axis=-1)
        channel2 = np.concatenate((X[:,:,6:9], X[:,:,9:12]), axis=-1)
        channel3 = np.concatenate((X[:,:,12:15], X[:,:,15:18]), axis=-1)
        channel4 = np.concatenate((X[:,:,18:21], X[:,:,21:24]), axis=-1)
        channel5 = np.concatenate((X[:,:,24:27], X[:,:,27:30]), axis=-1)
        all_X = np.stack([channel1, channel2, channel3, channel4, channel5], axis=1)  # [N, 5, T, 6]
        original_sampling_rate = 50
        num_classes = 8
    
    n_samples, n_channels, seq_len, n_features = all_X.shape
    all_X = all_X.reshape(n_samples * n_channels, seq_len, n_features)

    real_labels = real_labels.repeat_interleave(n_channels)

    # resample real data to 20 Hz
    new_sampling_rate = 20
    new_length = int((all_X.shape[1] / original_sampling_rate) * new_sampling_rate)
    resampled_data = np.array([resample(sequence, new_length) for sequence in all_X])


    if resampled_data.shape[1] < padding_size:
        resampled_data = np.pad(resampled_data, ((0, 0), (0, padding_size - resampled_data.shape[1]), (0, 0)), 'wrap') # N, 200, 6
    real_inputs = torch.from_numpy(resampled_data[:,:padding_size,:]).float()  

    if mode == 'train' and k and k < len(real_inputs):
        real_inputs, real_labels = select_samples(real_inputs, real_labels, k, dataset, data_path)
    print(f"{dataset}-SHAPE: {real_inputs.shape}, {real_labels.shape}")

    # load text
    label_dictionary = data['label_dictionary']
    label_list = [' '.join(labels) for labels in label_dictionary.values()]
    all_text = clip.tokenize(label_list).cuda()
    
    return real_inputs, real_labels, label_list, all_text, num_classes

def cross_load(dataset, padding_size, data_path, mode='test', k=None):

    X = np.load(f'{data_path}/{dataset}/X_{mode}.npy')
    real_labels = torch.from_numpy(np.load(f'{data_path}/{dataset}/y_{mode}.npy'))
    with open(f'{data_path}/{dataset}/{dataset}.json', 'r') as file:
        data = json.load(file)
    all_X = np.zeros((X.shape[0], X.shape[1], 22, 6))

    if dataset == 'DSADS':
        if mode == 'train':
            all_X[:,:,11] = np.concatenate((X[:,:,0:3], X[:,:,3:6]), axis=-1) # torso
            all_X[:,:,21] = np.concatenate((X[:,:,9:12], X[:,:,12:15]), axis=-1) # right arm
            all_X[:,:,17] = np.concatenate((X[:,:,18:21], X[:,:,21:24]), axis=-1) # left arm
            all_X[:,:,6] = np.concatenate((X[:,:,27:30], X[:,:,30:33]), axis=-1) # right leg
            all_X[:,:,2] = np.concatenate((X[:,:,36:39], X[:,:,39:42]), axis=-1) # left leg
        else:
            all_X[:,:,11] = np.concatenate((X[:,:,0:3], X[:,:,3:6]), axis=-1) 
            all_X[:,:,21] = np.concatenate((X[:,:,9:12], X[:,:,12:15]), axis=-1) 
            all_X[:,:,17] = np.concatenate((X[:,:,18:21], X[:,:,21:24]), axis=-1)
            all_X[:,:,6] = np.concatenate((X[:,:,27:30], X[:,:,30:33]), axis=-1)
            all_X[:,:,2] = np.concatenate((X[:,:,36:39], X[:,:,39:42]), axis=-1)
        original_sampling_rate = 25
        num_classes = 19
    
    all_X = all_X.reshape(all_X.shape[0], all_X.shape[1], 22 * 6)

    # resample real data to 20 Hz
    new_sampling_rate = 20
    new_length = int((all_X.shape[1] / original_sampling_rate) * new_sampling_rate)
    resampled_data = np.array([resample(sequence, new_length) for sequence in all_X])


    if resampled_data.shape[1] < padding_size:
        resampled_data = np.pad(resampled_data, ((0, 0), (0, padding_size - resampled_data.shape[1]), (0, 0)), 'wrap') # N, 200, 6
    real_inputs = torch.from_numpy(resampled_data[:,:padding_size,:]).float()  

    if mode == 'train' and k and k < len(real_inputs):
        real_inputs, real_labels = select_samples(real_inputs, real_labels, k, dataset, data_path)
    b, t, _ = real_inputs.shape
    real_inputs = real_inputs.reshape(b, t, 22, -1).permute(0, 3, 1, 2).unsqueeze(-1)
    real_inputs = real_inputs[:,:3,:,:]
    # nonzero_mask = (real_inputs != 0)
    # first_nonzero_indices = nonzero_mask.float().argmax(dim=3)
    # first_index_value = first_nonzero_indices[0, 0, 0].item()
    print(f"{dataset}-SHAPE: {real_inputs.shape}, {real_labels.shape}]")
 
    # load text
    label_dictionary = data['label_dictionary']
    label_list = [' '.join(labels) for labels in label_dictionary.values()]
    all_text = clip.tokenize(label_list).cuda()
    
    return real_inputs, real_labels, label_list, all_text, num_classes

def load_multiple(dataset_list, data_path, padding_size=200, mode='test', k=None, load_method="split"):

    real_inputs_list, real_labels_list, label_list_list, all_text_list, num_classes_list = [], [], [], [], []
    for dataset in dataset_list:
        if load_method == "split":
            real_inputs, real_labels, label_list, all_text, num_classes = split_load(dataset, padding_size, data_path, mode, k)
        elif load_method == "combine":
            real_inputs, real_labels, label_list, all_text, num_classes = combine_load(dataset, padding_size, data_path, mode, k)
            # real_inputs, real_labels, label_list, all_text, num_classes = cross_load(dataset, padding_size, data_path, mode, k)
        else:
            raise ValueError(f"Invalid load_method: {load_method}. Must be 'split' or 'combine'.")
        
        real_inputs_list.append(real_inputs)
        real_labels_list.append(real_labels)
        label_list_list.append(label_list)
        all_text_list.append(all_text)
        num_classes_list.append(num_classes)

    return real_inputs_list, real_labels_list, label_list_list, all_text_list, num_classes_list