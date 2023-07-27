import warnings
warnings.simplefilter("ignore")

original_filterwarnings = warnings.filterwarnings
def _filterwarnings(*args, **kwargs):
    return original_filterwarnings(*args, **{**kwargs, 'append':True})
warnings.filterwarnings = _filterwarnings

import os, sys
sys.path.append("../")
from utils import *
from models import *
from dataloader import *
import pickle
import params, json
import torchvision.models as models
import torch
import copy
from train import MyLogger

from layer_groups import *

def get_all_model_weights(model):
    all_params = [l for l in model.named_parameters()]
    all_params_dict = {}
    for tup in all_params:
        all_params_dict[tup[0]] = tup[1].cpu().detach()
    return all_params_dict


def train_mask_epoch(model, loader, optimizer, scheduler, mask= None):
    #This function is used to train the model on a subset of the data based on the mask
    #if mask is None, then train on all the data
    #else train on the data where mask is 1

    model.train()
    corrects, n, losses = 0, 0, 0
    for (x,y,ids) in loader:
        if mask is not None: 
            mask_ids = mask[ids]
            x,y = x[mask_ids==1].cuda(), y[mask_ids==1].cuda()
            # if mask_ids.sum() == 0 or 1 then skip the batch
            if mask_ids.sum() == 0 or mask_ids.sum() == 1:
                continue
        else:
            x, y = x.cuda(), y.cuda()
        
        preds = (model(x))
        corrects += (preds.argmax(1)==y).sum()
        n += y.shape[0]

        loss = torch.nn.CrossEntropyLoss()(preds, y)

        loss.backward()
        optimizer.step()
        if scheduler is not None: scheduler.step()
        losses += loss.detach()
        optimizer.zero_grad()
    
    return (corrects/n).detach().item(), (losses/n).detach().item()

def evaluate_mask(model, loader, mask = None):
    #This function is used to evaluate the model on a subset of the data based on the mask
    #if mask is None, then evaluate on all the data
    #else evaluate on the data where mask is 1
    model.eval()
    corrects, n = 0, 0
    for (x,y,ids) in loader:
        if mask is not None: 
            mask_ids = mask[ids]
            x,y = x[mask_ids==1].cuda(), y[mask_ids==1].cuda()
            # if mask_ids.sum() == 0 or 1 then skip the batch
            if mask_ids.sum() == 0 or mask_ids.sum() == 1:
                continue
        else:
            x, y = x.cuda(), y.cuda()
        preds = (model(x))
        corrects += (preds.argmax(1)==y).sum()
        n += y.shape[0]
    return corrects/n



def do_ft(model, train_loader, noise_mask, train_clean = True):
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler, EPOCHS = get_scheduler_epochs("triangle", optimizer, train_loader, max_epochs = 20)

    clean_accs, noisy_accs = [], []

    patience = 0

    # import ipdb; ipdb.set_trace()
    for ep in range(EPOCHS):
        if ep == 0:
            clean_acc = evaluate_mask(model, train_loader, 1 - noise_mask)
            noisy_acc = evaluate_mask(model, train_loader, noise_mask)
            clean_accs.append(clean_acc); noisy_accs.append(noisy_acc)
            print(f"Epoch: -1 | Clean Accuracy {clean_acc:.4f} | Noisy Accuracy {noisy_acc:.4f}")
        
        acc, loss = train_mask_epoch(model, train_loader, optimizer, scheduler, 1 - noise_mask if train_clean else noise_mask)
        
        clean_acc = evaluate_mask(model, train_loader, 1 - noise_mask)
        noisy_acc = evaluate_mask(model, train_loader, noise_mask)
        clean_accs.append(clean_acc); noisy_accs.append(noisy_acc)

        print(f"Epoch: {ep} | Clean Accuracy {clean_acc:.4f} | Loss {loss: .4f} | Noisy Accuracy {noisy_acc:.4f}")
        if acc == 1:
            patience += 1
        if patience >= 3:
            break
    
    return clean_accs, noisy_accs


def train_model(model, train_loader):
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler, EPOCHS = get_scheduler_epochs("triangle", optimizer, train_loader, max_epochs = 25)


    for ep in range(EPOCHS):
        rets = train_mask_epoch(model, train_loader, optimizer, scheduler, mask = None)
        print("Epoch: ", ep, "Train Acc: ", rets[0], "Train Loss: ", rets[1])

    return model

def layer_retraining(args, filename):
    pre_dict, ft_dict = return_loaders(args, get_frac = False, aug=args["augmentation"], cscore=args["cscore"])
    
    # with open(f'{filename}models.pickle', 'rb') as handle:
    #     all_models = pickle.load(handle)
    
    loader = pre_dict["train_loader"]
    noise_mask = pre_dict["noise_mask"]

    model_init = get_model(args)
    model = copy.deepcopy(model_init)

    # all_evals = []
    model_final = train_model(model, loader)

    #we need just the first and last models for this experiment
    # model_final.load_state_dict(all_models[-1])
    # model_init.load_state_dict(all_models[0])

    all_params_dict_init = get_all_model_weights(model_init)
    weight_groups = {"vit":vit_groups, "resnet50":resnet50_groups,"resnet9":resnet9_groups}[args["model_type"]]

    result_dict = {"noisy_training":{}, "clean_training":{}}

    for index in weight_groups:
        model = copy.deepcopy(model_final)
        params_new = model.state_dict()
        #set requires_grad to False for all the layers
        for key in params_new:
            params_new[key].requires_grad = False

        for key in weight_groups[index]:
            print(key)
            with torch.no_grad():
                params_new[key] = all_params_dict_init[key]
                #set requires_grad to True for the layers that we want to retrain
                params_new[key].requires_grad = True
            # break
        params_new_clean = copy.deepcopy(params_new)
        # params_new_noisy = copy.deepcopy(params_new)
        model.load_state_dict(params_new_clean)
        


        # model = copy.deepcopy(model_final)
        # for (name, param), (param_init) in zip(model.named_parameters(), model_init.parameters()):
        #     if name not in weight_groups[index]:
        #         param.requires_grad = False
        #     else:
        #         with torch.no_grad():
        #             #initialize with the starting model weights
        #             param.data = param_init.data

        # import ipdb; ipdb.set_trace()
        # z = evaluate_mask(model, loader, 1 - noise_mask)
        
        
        clean_training = do_ft(model, loader, noise_mask = 1 - noise_mask, train_clean = True)
        
        # model.load_state_dict(params_new_noisy)
        # model = copy.deepcopy(model_final)
        # for (name, param), (param_init) in zip(model.named_parameters(), model_init.parameters()):
        #     if name not in weight_groups[index]:
        #         param.requires_grad = False
        #     else:
        #         with torch.no_grad():
        #             #initialize with the starting model weights
        #             param.data = param_init.data

        # noisy_training = do_ft(model, loader, noise_mask = noise_mask, train_clean = False)
        
        # result_dict["noisy_training"][index] = noisy_training
        result_dict["clean_training"][index] = clean_training


    with open(f'{filename}reatraining.pickle', 'wb') as handle:
        pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return 0

def dir_to_args(args):
    # f = logs/svhn/resnet50_lr_0.001_noise_0.05_resnet50_cosine_seed_6_aug_1
    f = args["from_dir_name"]
    print(f)
    args["dataset1"] = f.split("/")[1]

    f_remaining = f.split("/")[2]
    #f_remaiing = resnet50_lr_0.001_noise_0.05_resnet50_cosine_seed_6_aug_1

    args["model_type"] = f_remaining.split("_")[0]
    args["lr1"] = float(f_remaining.split("_")[2])
    args["noise_1"] = float(f_remaining.split("_")[4])
    args["sched"] = f_remaining.split("_")[6]
    args["seed"] = int(f_remaining.split("_")[8])
    args["augmentation"] = int(f_remaining.split("_")[10])
    try:
        args["cscore"] = float(f_remaining.split("_")[12])
    except:
        args["cscore"] = 0.0
    return args

if __name__ == "__main__":
    parser = params.parse_args()
    args = parser.parse_args()
    args = params.add_config(args) if args.config_file != None else args
    args = vars(args)
    if args["from_dir_name"] is not None: args = dir_to_args(args)
    args["dataset2"] = args["dataset1"]
    args["num_epochs"] = 100 if args["augmentation"] else 50
    args["num_epochs"] = 100 if args["dataset1"] == "mnist" else args["num_epochs"]
    if args["model_type"] == "vit": args["batch_size"] = 128
    print(args)
    filename = f'../logs/{args["dataset1"]}/{args["model_type"]}_lr_{args["lr1"]}_noise_{args["noise_1"]}_{args["model_type"]}_{args["sched"]}_seed_{args["seed"]}_aug_{args["augmentation"]}_cscore_{args["cscore"]}/'
    if not (os.path.exists(filename)):
        os.makedirs(filename)
    
    output_pickle = f'{filename}retraining.pickle'
    if (os.path.exists(output_pickle)):
        print("Already exists")
        exit(0)

    seed_everything(args["seed"])

    # sys.stdout = MyLogger(f"{filename}/retraining.log", "a")
    layer_retraining(args, filename)