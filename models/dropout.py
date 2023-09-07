from models import *
import torch
import numpy as np

class FixedDropout(torch.nn.Module):
    def __init__(self, p, drop_mode = "train"):
        super(FixedDropout, self).__init__()
        # Implements Sparse dropout. This essentially simulates a neural network with lower capacity.
        # We deterministically 0 out the same neurons at train and test time
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
        # Implements Standard dropout. 
        self.p = p
        self.drop_mode = drop_mode

    def forward(self, x, idx, epoch = 0):
        if self.drop_mode != "train":
            return x
        mask = (torch.rand_like(x) > self.p).float() # Random mask with dropout probability
        x = x * mask # Apply dropout mask
        x = x / (1 - self.p) # Scale up the output to compensate for the dropped neurons
        return x



class ExampleTiedDropout(torch.nn.Module):
    # this class is similar to batch tied dropout, 
    # but instead of tying neurons in a batch, we tie neurons in a set of examples

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
                    # In an initial implementation, rather than repeating the same mask along all dimensions,
                    # the following was done. This meant that we will randomly have 0s and 1s along different dimensions
                    # removed this so that it preserves image pixel semantics.

                    # mask[i][int(self.p_fixed*shape):] = torch.bernoulli(torch.full((1, shape_of_mask, X.shape[2], X.shape[3]), p_mem))
                    

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

