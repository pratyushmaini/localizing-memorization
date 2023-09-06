from models import *
import torch
import numpy as np

class FixedDropout(torch.nn.Module):
    def __init__(self, p, drop_mode = "train"):
        super(FixedDropout, self).__init__()
        #implements sparse dropout. We deterministically 0 out the same neurons at train and test time
        self.p = p
        self.drop_mode = drop_mode

    def forward(self, x, idx, epoch = 0):
        torch.manual_seed(0)
        mask = (torch.rand_like(x) > self.p).float() # Random mask with dropout probability
        x = x * mask # Apply dropout mask
        x = x / (1 - self.p) # Scale up the output to compensate for the dropped neurons
        return x
    
class StandardDropout(torch.nn.Module):
    def __init__(self, p, drop_mode = "train"):
        super(StandardDropout, self).__init__()
        self.p = p
        self.drop_mode = drop_mode

    def forward(self, x, idx, epoch = 0):
        if self.drop_mode != "train":
            return x
        mask = (torch.rand_like(x) > self.p).float() # Random mask with dropout probability
        x = x * mask # Apply dropout mask
        x = x / (1 - self.p) # Scale up the output to compensate for the dropped neurons
        return x



class BatchTiedDropout(torch.nn.Module):
    def __init__(self, p_fixed = 0.2, p_mem = 0.1, num_batches = 100, drop_mode = "train"):
        super(BatchTiedDropout, self).__init__()
        
        self.p_mem = p_mem
        self.p_fixed = p_fixed
        self.num_batches = num_batches
        self.num_sets = (1-p_fixed)/p_mem
        self.num_repeats = num_batches/self.num_sets
        self.drop_mode = drop_mode

        #with every batch of examples, tie a small fraction of the neurons together
        #we will keep a fraction of neurons always fixed. lets call this fixed_p = 0.2 (defualt)
        #let us say there are 100 batches in the training set = num_batches.
        #the remaining 0.8 fraction of neurons will be randomly tied together in each batch = 1-fixed_p
        #let us allocate 0.1 fraction of neurons to each batch = p_mem. so there are 8 sets of neuron ids : num_sets = (1-fixed_p)/p_mem
        #so each neuron set will occur in num_repeats = (num_batches/num_sets) batches

    def forward(self, X, batch_id):

        if self.drop_mode == "train":
            #At training time, we use index_fraction to decide which neurons to keep
            
            #number of dimensions
            shape = X.shape[1]
            mask = torch.zeros_like(X)

            #keep all neurons with index less than self.p_fixed*shape
            mask[:, :int(self.p_fixed*shape)] = 1

            #use the index fraction to decide which neurons to keep by using modulus over the remaining neurons
            #first find the index of the neuron set to keep
            remaining_fraction = 1 - self.p_fixed
            index_start_id = int(self.p_fixed*shape) + ((batch_id*int(self.p_mem*shape))% int(remaining_fraction*shape))

            #now keep all neurons with index less than p_mem*shape
            mask[:, index_start_id : index_start_id + int(self.p_mem*shape)] = 1
    
            X = X*mask
            
        elif self.drop_mode == "test":
            #At test time we will renormalize outputs from the non-fixed neurons based on the number of neuron sets
            #we will keep the fixed neurons unmodified
            shape = X.shape[1]
            X[:, :int(self.p_fixed*shape)] = X[:, :int(self.p_fixed*shape)]
            X[:, int(self.p_fixed*shape):] = X[:, int(self.p_fixed*shape):]/self.num_sets

        elif self.drop_mode == "drop":
            shape = X.shape[1]
            X[:, int(self.p_fixed*shape):] = 0
            X[:, :int(self.p_fixed*shape)] = X[:, :int(self.p_fixed*shape)]*(self.p_fixed + self.p_mem)/self.p_fixed
        return X

class ExampleTiedDropout(torch.nn.Module):
    # this class is similar to batch tied dropout, but instead of tying neurons in a batch, we tie neurons in a set of examples

    def __init__(self, p_fixed = 0.2, p_mem = 0.1, num_batches = 100, drop_mode = "train"):
        super(ExampleTiedDropout, self).__init__()
        self.seed = 101010
        self.max_id = 60000
        self.p_mem = p_mem
        self.p_fixed = p_fixed
        self.drop_mode = drop_mode
        self.mask_tensor = None

    def forward(self, X, idx, epoch = 0):
        if self.p_fixed == 1:
            return X

        if self.drop_mode == "train":
            # create a mask based on the index (idx)

            mask = torch.zeros_like(X).cpu()
            shape = X.shape[1]

            if epoch > 0:
                #get mask from self.mask_tensor
                mask = self.mask_tensor[idx]

            elif epoch == 0:
                #keep all neurons with index less than self.p_fixed*shape
                mask[:, :int(self.p_fixed*shape)] = 1

                # Fraction of elements to keep
                p_mem = self.p_mem

                # Generate a random mask for each row in the input tensor
                shape_of_mask = shape - int(self.p_fixed*shape)
                for i in range(X.shape[0]):
                    torch.manual_seed(idx[i].item())
                    curr_mask = torch.bernoulli(torch.full((1, shape_of_mask), p_mem))
                    #repeat curr_mask along dimension 2 and 3 to have the same shape as X
                    curr_mask = curr_mask.unsqueeze(-1).unsqueeze(-1)
                    mask[i][int(self.p_fixed*shape):] = curr_mask
                    # mask[i][int(self.p_fixed*shape):] = torch.bernoulli(torch.full((1, shape_of_mask, X.shape[2], X.shape[3]), p_mem))
                    # mask[i] = torch.bernoulli(torch.full((1, X.shape[1], X.shape[2], X.shape[3]), p_mem))

                # import ipdb; ipdb.set_trace()
                if self.mask_tensor is None:
                    self.mask_tensor = torch.zeros(self.max_id, X.shape[1], X.shape[2], X.shape[3])
                #assign mask at positions given by idx
                self.mask_tensor[idx] = mask
            
            # Apply the mask to the input tensor
            X = X * mask.cuda()


            
        elif self.drop_mode == "test":
            #At test time we will renormalize outputs from the non-fixed neurons based on the number of neuron sets
            #we will keep the fixed neurons unmodified
            shape = X.shape[1]
            X[:, :int(self.p_fixed*shape)] = X[:, :int(self.p_fixed*shape)]
            X[:, int(self.p_fixed*shape):] = X[:, int(self.p_fixed*shape):]*self.p_mem

        elif self.drop_mode == "drop":
            shape = X.shape[1]
            X[:, int(self.p_fixed*shape):] = 0
            X[:, :int(self.p_fixed*shape)] = X[:, :int(self.p_fixed*shape)]*(self.p_fixed + self.p_mem)/self.p_fixed
        
        return X


class ExampleTiedDropout2(torch.nn.Module):
    # this class is similar to batch tied dropout, but instead of tying neurons in a batch, we tie neurons in a set of examples

    def __init__(self, p_fixed = 0.2, p_mem = 0.1, num_batches = 100, drop_mode = "train"):
        super(ExampleTiedDropout2, self).__init__()
        self.seed = 101010
        self.max_id = 60000
        self.p_mem = p_mem
        self.p_fixed = p_fixed
        self.drop_mode = drop_mode
        self.mask_tensor = None

    def forward(self, X, idx, epoch = 0):
        if self.p_fixed == 1:
            return X

        if self.drop_mode == "train":
            # create a mask based on the index (idx)

            mask = torch.zeros_like(X).cpu()
            shape = X.shape[1]

            if epoch > 0:
                #get mask from self.mask_tensor
                mask = self.mask_tensor[idx]

            elif epoch == 0:
                #keep all neurons with index less than self.p_fixed*shape
                mask[:, :int(self.p_fixed*shape)] = 1

                # Fraction of elements to keep
                p_mem = self.p_mem

                # Generate a random mask for each row in the input tensor
                shape_of_mask = shape - int(self.p_fixed*shape)
                for i in range(X.shape[0]):
                    torch.manual_seed(idx[i].item())
                    mask[i][int(self.p_fixed*shape):] = torch.bernoulli(torch.full((1, shape_of_mask, X.shape[2], X.shape[3]), p_mem))

                # import ipdb; ipdb.set_trace()
                if self.mask_tensor is None:
                    self.mask_tensor = torch.zeros(self.max_id, X.shape[1], X.shape[2], X.shape[3])
                #assign mask at positions given by idx
                self.mask_tensor[idx] = mask
            
            # Apply the mask to the input tensor
            X = X * mask.cuda()


            
        elif self.drop_mode == "test":
            #At test time we will renormalize outputs from the non-fixed neurons based on the number of neuron sets
            #we will keep the fixed neurons unmodified
            shape = X.shape[1]
            X[:, :int(self.p_fixed*shape)] = X[:, :int(self.p_fixed*shape)]
            X[:, int(self.p_fixed*shape):] = X[:, int(self.p_fixed*shape):]*self.p_mem

        elif self.drop_mode == "drop":
            shape = X.shape[1]
            X[:, int(self.p_fixed*shape):] = 0
            X[:, :int(self.p_fixed*shape)] = X[:, :int(self.p_fixed*shape)]*(self.p_fixed + self.p_mem)/self.p_fixed
        
        return X