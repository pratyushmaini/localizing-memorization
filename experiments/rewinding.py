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

def eval_acc_change_in_all_weights(args, all_models, loader, reinit_epoch = 0):
    model_init = get_model(args)
    model_final = get_model(args)

    all_evals = []
    
    #we need just the first and last models for this experiment
    model_final.load_state_dict(all_models[-1])
    if reinit_epoch != -1:
        model_init.load_state_dict(all_models[reinit_epoch])

    all_params_dict_init = get_all_model_weights(model_init)

    
    weight_groups = {"vit":vit_groups, "resnet50":resnet50_groups,"resnet9":resnet9_groups}[args["model_type"]]


    for index in weight_groups:
        new_model = copy.deepcopy(model_final)
        params_new = new_model.state_dict()
        for key in weight_groups[index]:
            with torch.no_grad():
                params_new[key] = all_params_dict_init[key]

        new_model.load_state_dict(params_new)
        eval_ret =  eval(new_model, loader, eval_mode = True)
        all_evals.append(eval_ret)
    
    return all_evals

def layer_rewinding(args, filename):
    pre_dict, ft_dict = return_loaders(args, get_frac = False, aug=args["augmentation"])
    
    with open(f'{filename}models.pickle', 'rb') as handle:
        all_models = pickle.load(handle)
    
    loader = pre_dict["train_loader"]
    ep_all_evals = []
    
    # for ep in range(0,len(all_models),10):
    for ep in range(0,15,2):
        all_evals = eval_acc_change_in_all_weights(args, all_models, loader, reinit_epoch = ep)
        ep_all_evals.append(all_evals)

    with open(f'{filename}rewinding.pickle', 'wb') as handle:
        pickle.dump(ep_all_evals, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
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
    
    
    assert (os.path.exists(filename))

    output_pickle = f'{filename}rewinding.pickle'
    # if (os.path.exists(output_pickle)):
    #     print("Already exists")
    #     exit(0)

    seed_everything(args["seed"])

    sys.stdout = MyLogger(f"{filename}/rewinding.log", "a")
    layer_rewinding(args, filename)