from torch import nn
from tqdm import tqdm
import copy
import torch
import sys
sys.path.append("../")
from utils import *
from multiprocessing.pool import ThreadPool
import itertools
import ipdb
import pickle


def unravel_index(index, shape):
    #torch.argmax returns index for a flattened tensor. to be able to index it later on we need to unravel it.
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))

def get_population_loss(loader, model):
    loss = 0
    grads_list = {}
    for x,y,ids in loader:
        loss = nn.CrossEntropyLoss(reduction = "mean")(model(x.cuda()), y.cuda().long())
        loss.backward()
        for name,param in (model.named_parameters()):
            if name not in grads_list:
                grads_list[name] = -1*param.grad.detach()
            else:
                grads_list[name] -= param.grad.detach()
        
    for name,param in (model.named_parameters()):
        grads_list[name] /= len(loader)
    model.zero_grad()
    return grads_list


def l2_noise(inp, norm_ratio):
    '''
    inp: input on which noise is added
    norm_ratio: ratio of norm of noise to that of input
    '''
    noise_2 = torch.normal(0, 1, size=inp.shape).cuda()
    noise_2 *= norm_ratio*inp.norm()/noise_2.norm()
    return noise_2


def old_objective():
    grads_list_population = get_population_loss(loader, saved_model)
    loss = 0.1*nn.CrossEntropyLoss()(preds, n_y.long())
    grads_list = copy.deepcopy(grads_list_population)
    loss.backward()
    for name,param in (model.named_parameters()):
        grads_list[name] += param.grad.detach()


def get_new_grads(model, x,y,current_example_index, robustify = False, n_EoT = 1):
    '''
    robustify: To get robust estimate of gradients, should we add gaussian noise to input 
    n_EoT: number of steps for Expectation over transformation (gaussian noise)
    returns grads_list: dictionary of gradients corresponding to each parameter in the model
    '''
    grads_list = {}
    final_preds = None
    n_EoT = 1 if not robustify else n_EoT
    for _ in range (n_EoT):
        # import ipdb; ipdb.set_trace()
        if robustify:
            # To get robust estimate of gradients, we will add gaussian noise to sample
            x = x + l2_noise(x, 0.01)

        preds = model(x)    
        final_preds = preds.detach() if final_preds is None else final_preds + preds.detach()

        loss = nn.CrossEntropyLoss(reduction = 'none')(preds, y)
        batch_size = y.shape[0]
        #for the example that we want to flip, we must reverse the loss while maintaing the population loss
        loss[current_example_index] *= -1*batch_size
        loss = loss.mean()
        loss.backward()

        for name,param in (model.named_parameters()):
            if name in grads_list: grads_list[name] += param.grad.detach()
            else: grads_list[name] = copy.deepcopy(param.grad.detach())

        # ipdb.set_trace()
        model.zero_grad()

    for name,param in (model.named_parameters()):
        grads_list[name] /= n_EoT

    
    return grads_list, preds/n_EoT


def get_most_activated_node(model, grads_list, channel_wise = "channel", objective = "zero"):
    '''
    channel wise: Remove weights at the channel level versus at the neuron level
    '''
    max_val = 0
    max_param_name = None
    max_param_index = None
    for name,param in (model.named_parameters()):
        if objective == "zero":
            signed_grad = (param.data*grads_list[name])
        else:
            assert(objective == "step")
            signed_grad = grads_list[name].abs()
        
        if len(param.data.shape)==4 and channel_wise == "channel":
            # is this a conv head (channel wise)
            signed_grad = signed_grad.sum(dim=(1,2,3))
        signed_max = signed_grad.max()  

        if signed_max>max_val:
            max_val = signed_max
            max_param_name = name
            max_param_index = unravel_index(signed_grad.argmax(), signed_grad.shape)
    
    return max_val, max_param_name, max_param_index

def modify_weights(model, max_param_name, max_param_index, channel_wise = "channel", objective = "zero", grads_list = None, alpha = 1, preds = None):
    with torch.no_grad():
        for name,param in (model.named_parameters()):
            if name != max_param_name: continue
            # the index will automatically take care of node versus channel wise. 
            # if channel wise, it should be a single integer. 
            # if node wise, and the parameter is conv layer, then it should be a tuple to the exact neuron
            
            if objective == "zero": 
                param[max_param_index] = 0
            else:
                assert(objective == "step")
                # ipdb.set_trace()
                param[max_param_index] -= alpha*grads_list[name][max_param_index]
                print(name, max_param_index[0].item(), alpha*grads_list[name][max_param_index].item(), param[max_param_index].item(), preds[0][preds[1]].item(), preds[1])

    return model


def flip_preds_loop_helper(saved_model, loader, batch, indices, batch_mask, eval_post_edit, channel_wise, objective, gaussian_noise, verbose, n_EoT, noise_mask):
    accs_list, iters_list, params_list, ids_list = [], [], [], []
    noisy_acc_list, clean_acc_list = [], []
    all_ids = torch.arange(batch[0].shape[0])
    # ipdb.set_trace()
    for num_examples_analyzed in tqdm(indices):
        current_example_index = all_ids[batch_mask == 1][num_examples_analyzed]
        original_id = batch[2][current_example_index]

        ## Minimum set of weights that can be changed
        model = copy.deepcopy(saved_model)
        model.eval()
        iters = 0
        param_names = []
        while True:
            grads_list, preds = get_new_grads(model, batch[0].cuda(), batch[1].cuda().long(),current_example_index, robustify = gaussian_noise, n_EoT = n_EoT)
            if preds[current_example_index].argmax() != batch[1][current_example_index]:
                break
            iters +=1
            max_val, max_param_name, max_param_index = get_most_activated_node(model, grads_list, channel_wise = channel_wise)
            model = modify_weights(model, max_param_name, max_param_index, channel_wise = channel_wise, objective = objective, grads_list = grads_list, preds = (preds[current_example_index], batch[1][current_example_index]))
            param_names.append(max_param_name)
            
        if eval_post_edit:
            rets = eval(model, loader, eval_mode=True)
            rets["clean_accuracy"] = rets["acc_mask"][noise_mask == 0].mean()
            rets["noisy_accuracy"] = rets["acc_mask"][noise_mask == 1].mean()
            accs_list.append(rets["accuracy"])
            noisy_acc_list.append(rets["noisy_accuracy"])
            clean_acc_list.append(rets["clean_accuracy"])
        
        iters_list.append(iters)
        params_list.append(param_names)
        ids_list.append(original_id)
        if verbose:
            print("Iters", iters, rets["accuracy"], param_names)
    return accs_list, iters_list, params_list, ids_list, noisy_acc_list, clean_acc_list

def flip_preds(saved_model, loader, example_type, noise_mask, rare_mask = None, eval_post_edit=True, num_examples = 100, verbose = False, channel_wise = "channel", objective = "zero", gaussian_noise = False, n_EoT = 1, n_parallel = 1):
    '''
    saved_model: original model that is to be changed
    loader: data loader
    example_type: can be rare, noisy or clean
    '''
    
    accs_list, iters_list, params_list, ids_list = [], [], [], []
    noisy_acc_list, clean_acc_list = [], []

    clean_mask = ~(noise_mask.bool() + rare_mask.bool()) if rare_mask is not None else ~ (noise_mask.bool())
    mask = {"clean":clean_mask, "noisy":noise_mask, "rare":rare_mask}[example_type]

    
    num_examples_analyzed = 0

    while num_examples_analyzed < num_examples:
        batch = next(iter(loader))
        all_ids = torch.arange(batch[0].shape[0])
        batch_mask = mask[batch[2]]
       
        # Parallelize the for loop
        # how many valid indices does the batch have for the current example type?
        num_remaining = num_examples - num_examples_analyzed
        num_valid_datapoints_in_batch = min(all_ids[batch_mask == 1].shape[0],num_remaining)
        num_valid_datapoints_per_job = num_valid_datapoints_in_batch//n_parallel
        #split the data points across the threads
        indices = np.arange(num_valid_datapoints_in_batch)   
        ind_arglist = [indices[k*num_valid_datapoints_per_job:(k+1)*num_valid_datapoints_per_job] for k in range(n_parallel)]
        arglist = [(saved_model, loader, batch, ind, batch_mask, eval_post_edit, channel_wise, objective, gaussian_noise, verbose, n_EoT, noise_mask) for ind in ind_arglist]

        pool = ThreadPool()     #(initializer=init_processes, initargs=(globVar,))
        result = pool.starmap(flip_preds_loop_helper, arglist)
        # import ipdb; ipdb.set_trace()
        
        accs_list_temp, iters_list_temp, params_list_temp, ids_list_temp, n_acc_l_temp, c_acc_l_temp = [x[0] for x in result], [x[1] for x in result], [x[2] for x in result], [x[3] for x in result], [x[4] for x in result], [x[5] for x in result] 
        accs_list_temp, iters_list_temp, params_list_temp, ids_list_temp, n_acc_l_temp, c_acc_l_temp = concat_list(accs_list_temp), concat_list(iters_list_temp), concat_list(params_list_temp), concat_list(ids_list_temp), concat_list(n_acc_l_temp), concat_list(c_acc_l_temp)

        accs_list += accs_list_temp
        iters_list += iters_list_temp
        params_list += params_list_temp
        ids_list += ids_list_temp
        noisy_acc_list += n_acc_l_temp
        clean_acc_list += c_acc_l_temp
        pool.close()

        num_examples_analyzed += num_valid_datapoints_in_batch

    return {"accuracy":accs_list, "iters":iters_list, "params":params_list, "ids": ids_list, "noisy_acc":noisy_acc_list, "clean_acc":clean_acc_list}

def concat_list(a):
    return list(itertools.chain.from_iterable(a))


def rand_steps(model, X, y, args):
    #optimized implementation to only query remaining points
    is_training = model.training
    model.eval()                    # Need to freeze the batch norm and dropouts
    
    #Define the Noise
    uni, std, scale = (0.005, 0.005, 0.01); steps = 500
    noise_2 = lambda X: torch.normal(0, std, size=X.shape).cuda()
    noise_1 = lambda X: torch.from_numpy(np.random.laplace(loc=0.0, scale=scale, size=X.shape)).float().to(X.device) 
    noise_inf = lambda X: torch.empty_like(X).uniform_(-uni,uni)

    noise_map = {"l1":noise_1, "l2":noise_2, "linf":noise_inf}
    mag = 1

    delta = noise_map[args.distance](X)
    delta_base = delta.clone()
    delta.data = torch.min(torch.max(delta.detach(), -X), 1-X)  
    loss = 0
    with torch.no_grad():
        for t in range(steps):   
            if t>0: 
                preds = model(X_r+delta_r)
                new_remaining = (preds.max(1)[1] == y[remaining])
                remaining[remaining] = new_remaining
            else: 
                preds = model(X+delta)
                remaining = (preds.max(1)[1] == y)
                
            if remaining.sum() == 0: break

            X_r = X[remaining]; delta_r = delta[remaining]
            preds = model(X_r + delta_r)
            mag+=1; delta_r = delta_base[remaining]*mag
            delta_r.data = torch.min(torch.max(delta_r.detach(), -X_r), 1-X_r) # clip X+delta_r[remaining] to [0,1]
            delta[remaining] = delta_r.detach()
            
        print(f"Number of steps = {t+1} | Failed to convert = {(model(X+delta).max(1)[1]==y).sum().item()}")
    if is_training:
        model.train()    
    return 

def attribution_confs(saved_model, pre_dict, filename):
    loader = pre_dict["train_loader"]                                                 
    noise_mask = pre_dict["noise_mask"]                                                                                                                                       
    batch = next(iter(loader))                                                        
    preds = saved_model(batch[0].cuda()) 
    probs = nn.Softmax(1)(preds)                                             
    confs  = probs[torch.arange(batch[1].shape[0]), batch[1].cuda()]    
    batch_noise_mask = noise_mask[batch[2]]                                               
    clean_confs = confs[batch_noise_mask==0]     
    noisy_confs = confs[batch_noise_mask==1]     


    with open(f"{filename}noisy_confs.pickle", "wb") as output_file:
        pickle.dump(noisy_confs, output_file)

    with open(f"{filename}clean_confs.pickle", "wb") as output_file:
        pickle.dump(clean_confs, output_file)