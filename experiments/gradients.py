#This python program is used to plot the gradients of the loss function with respect to the parameters of the network

import matplotlib.pyplot as plt
import numpy as np
import pickle, os, sys
from layer_groups import *


def get_all_grads(trackables_filename):
    model_type = "resnet50" if "resnet50" in trackables_filename else "vit" if "vit" in trackables_filename else "resnet9"
    weight_groups = {"vit":vit_groups, "resnet50":resnet50_groups,"resnet9":resnet9_groups}[model_type]
    #trackables is a pickle file name. We will load that pickle file and return the gradients
    with open(trackables_filename, 'rb') as handle:
        trackables = pickle.load(handle)

    '''trackables.keys()
    Out[5]: dict_keys(['noise_mask', 'clean_total_1_grads_all_epochs', 'noisy_total_1_grads_all_epochs', 'clean_total_2_grads_all_epochs', 'noisy_total_2_grads_all_epochs', 'total_total_grads_all_epochs', 'clean_nr_grads_all_epochs', 'noisy_nr_grads_all_epochs', 'total_nr_grads_all_epochs', 'clean_dr_grads_all_epochs', 'noisy_dr_grads_all_epochs', 'total_dr_grads_all_epochs', 'total_loss_all_epochs', 'clean_total_1_loss_all_epochs', 'noisy_total_1_loss_all_epochs', 'clean_total_2_loss_all_epochs', 'noisy_total_2_loss_all_epochs', 'clean_nr_loss_all_epochs', 'noisy_nr_loss_all_epochs', 'clean_dr_loss_all_epochs', 'noisy_dr_loss_all_epochs', 'total_correct_all_epochs', 'clean_correct_all_epochs', 'noisy_correct_all_epochs', 'total_num_all_epochs', 'clean_num_all_epochs', 'noisy_num_all_epochs', 'eval_all_epochs'])
    '''


    # num_epochs = len(clean_total_1_grads_all_epochs) 
    num_epochs = len(trackables["clean_total_1_grads_all_epochs"])

    #len gg[0] = len(model.named_parameters()) 

    # we need to combine the gradients based on the param groups defined in "layer_groups.py"
    #we combine them by adding their individual norms. we will also normalize these norms by the number of parameters in the group

    #finally return a tensor of shape num_epochs * len(weight_groups) with each element containing the normalized gradient norms

    #weight_groups is a dictionary with keys as index of layer and values as the list of parameters in that layer
    #get all model parameters by combining list of all values in weight_groups
    model_parameters = [val for sublist in weight_groups.values() for val in sublist]
    print(model_parameters)

    tensor_of_norms_total = np.zeros((num_epochs, len(weight_groups)))
    tensor_of_norms_clean = np.zeros((num_epochs, len(weight_groups)))
    tensor_of_norms_noisy = np.zeros((num_epochs, len(weight_groups)))

    for epoch in range(num_epochs):
        clean_total_1_grads = trackables["clean_total_1_grads_all_epochs"][epoch]
        noisy_total_1_grads = trackables["noisy_total_1_grads_all_epochs"][epoch]
        total_total_grads = trackables["total_total_grads_all_epochs"][epoch]
        # clean_nr_grads = trackables["clean_nr_grads_all_epochs"][epoch]
        # noisy_nr_grads = trackables["noisy_nr_grads_all_epochs"][epoch]
        # total_nr_grads = trackables["total_nr_grads_all_epochs"][epoch]
        # clean_dr_grads = trackables["clean_dr_grads_all_epochs"][epoch]
        # noisy_dr_grads = trackables["noisy_dr_grads_all_epochs"][epoch]
        # total_dr_grads = trackables["total_dr_grads_all_epochs"][epoch]

        param_counter = 0
        for index in weight_groups:
            all_params = weight_groups[index]
            num_params = len(all_params)
            for i in range(num_params):
                c,n,t = clean_total_1_grads[param_counter], noisy_total_1_grads[param_counter], total_total_grads[param_counter]
                param_counter += 1
                tensor_of_norms_total[epoch, index] += np.linalg.norm(t)/np.sqrt(len(t))
                tensor_of_norms_clean[epoch, index] += np.linalg.norm(c)/np.sqrt(len(c))
                tensor_of_norms_noisy[epoch, index] += np.linalg.norm(n)/np.sqrt(len(n))

    return tensor_of_norms_total, tensor_of_norms_clean, tensor_of_norms_noisy

def get_all_grads_cosine_similarity(trackables_filename):
    model_type = "resnet50" if "resnet50" in trackables_filename else "vit" if "vit" in trackables_filename else "resnet9"
    weight_groups = {"vit":vit_groups, "resnet50":resnet50_groups,"resnet9":resnet9_groups}[model_type]
    #trackables is a pickle file name. We will load that pickle file and return the gradients
    with open(trackables_filename, 'rb') as handle:
        trackables = pickle.load(handle)

    '''trackables.keys()
    Out[5]: dict_keys(['noise_mask', 'clean_total_1_grads_all_epochs', 'noisy_total_1_grads_all_epochs', 'clean_total_2_grads_all_epochs', 'noisy_total_2_grads_all_epochs', 'total_total_grads_all_epochs', 'clean_nr_grads_all_epochs', 'noisy_nr_grads_all_epochs', 'total_nr_grads_all_epochs', 'clean_dr_grads_all_epochs', 'noisy_dr_grads_all_epochs', 'total_dr_grads_all_epochs', 'total_loss_all_epochs', 'clean_total_1_loss_all_epochs', 'noisy_total_1_loss_all_epochs', 'clean_total_2_loss_all_epochs', 'noisy_total_2_loss_all_epochs', 'clean_nr_loss_all_epochs', 'noisy_nr_loss_all_epochs', 'clean_dr_loss_all_epochs', 'noisy_dr_loss_all_epochs', 'total_correct_all_epochs', 'clean_correct_all_epochs', 'noisy_correct_all_epochs', 'total_num_all_epochs', 'clean_num_all_epochs', 'noisy_num_all_epochs', 'eval_all_epochs'])
    '''


    # num_epochs = len(clean_total_1_grads_all_epochs) 
    num_epochs = len(trackables["clean_total_1_grads_all_epochs"])

    
    model_parameters = [val for sublist in weight_groups.values() for val in sublist]
    print(model_parameters)

    tensor_cosine_similarity_total = np.zeros((num_epochs, len(weight_groups)))

    cosine_similarity = lambda x,y: np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

    for epoch in range(num_epochs):
        clean_total_1_grads = trackables["clean_total_1_grads_all_epochs"][epoch]
        noisy_total_1_grads = trackables["noisy_total_1_grads_all_epochs"][epoch]
        
        param_counter = 0
        for index in weight_groups:
            all_params = weight_groups[index]
            num_params = len(all_params)
            for i in range(num_params):
                c,n = clean_total_1_grads[param_counter], noisy_total_1_grads[param_counter]
                param_counter += 1
                tensor_cosine_similarity_total[epoch, index] += cosine_similarity(c.numpy(),n.numpy())
            
            tensor_cosine_similarity_total[epoch, index] /= num_params

    return tensor_cosine_similarity_total