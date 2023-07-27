import warnings
warnings.simplefilter("ignore")

original_filterwarnings = warnings.filterwarnings
def _filterwarnings(*args, **kwargs):
    return original_filterwarnings(*args, **{**kwargs, 'append':True})
warnings.filterwarnings = _filterwarnings


import os, sys, ipdb
from utils import *
from models import *
from dataloader import *
import pickle
import params, json
import torchvision.models as models
import torch
import torch.nn as nn

class MyLogger(object):
    def __init__(self, filename="default.log", mode="a"):
        self.terminal = sys.stdout
        self.log = open(filename, mode)
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass    

def grads_to_norms(trackables):
    for key in trackables.keys():
        if "grads" in key:
            for i in range(len(trackables[key])):
                trackables[key][i] = trackables[key][i].norm()
    return trackables

def update_gradients(grads, model):
    if grads == []:
        #First iteration in batch
        for param in model.parameters():
            grads.append(param.grad.view(-1).detach().cpu())
    else:
        for i,param in enumerate(model.parameters()):
            grads[i] += (param.grad.view(-1).detach().cpu())
    
    # for param in model.parameters():
    #     param.grad = None
    #take the norm of grads and divide by the param square
    # grads_norm = grads.norm()
    return grads

def assert_no_grads(model):
    for param in model.parameters():
        assert(param.grad is None or param.grad.abs().max() == 0)


def get_trackables_single_epoch():
    trackables = {}
    trackables[f'clean_total_1_grads'] = []
    trackables[f'noisy_total_1_grads'] = []
    trackables[f'clean_total_2_grads'] = []
    trackables[f'noisy_total_2_grads'] = []
    trackables[f'total_total_grads'] = []

    trackables['clean_nr_grads'] = []
    trackables['noisy_nr_grads'] = []
    trackables['total_nr_grads'] = []

    trackables['clean_dr_grads'] = []
    trackables['noisy_dr_grads'] = []
    trackables['total_dr_grads'] = []

    trackables['total_loss'], trackables['clean_total_1_loss'], trackables['noisy_total_1_loss'] = 0, 0, 0
    trackables['total_loss'], trackables['clean_total_2_loss'], trackables['noisy_total_2_loss'] = 0, 0, 0
    
    trackables['clean_nr_loss'], trackables['noisy_nr_loss'] = 0, 0
    trackables['clean_dr_loss'], trackables['noisy_dr_loss'] = 0, 0
   
    trackables['total_correct'], trackables['clean_correct'], trackables['noisy_correct'] = 0, 0, 0
    trackables['total_num']  = 0
    trackables['clean_num']  = 0
    trackables['noisy_num']  = 0

    return trackables

def update_trackables(model, optimizer, loss_fn, preds, targets, ratio, trackables, name):
    loss_clean = ratio*loss_fn(preds, targets)
    trackables[f'{name}_loss'] += loss_clean.cpu().item()
    loss_clean.backward(retain_graph=True)
    trackables[f'{name}_grads'] = update_gradients(trackables[f'{name}_grads'], model)
    optimizer.zero_grad()
    return trackables

def single_epoch(model, optimizer, loader, loss_fn, scheduler, noise_mask = None, track_gradients = False, epoch = 0):
    trackables = get_trackables_single_epoch()
    clean_num, noisy_num = noise_mask.shape[0] - noise_mask.sum(), noise_mask.sum()
    for ims, labs, ids in loader:
        optimizer.zero_grad(set_to_none=True)
        ims, labs = ims.cuda(), labs.cuda()
        
        out = model(ims)
        noisy_ids = noise_mask[ids]
        clean_ids = 1 - noisy_ids
        
        clean_ratio = clean_ids.sum()/clean_ids.shape[0]
        noisy_ratio = 1 - clean_ratio
        
        # Clean Examples
        preds, targets, ratio = out[clean_ids==1], labs[clean_ids==1], clean_ratio
        
        ## get full grads
        trackables = update_trackables(model, optimizer, loss_fn, preds, targets, ratio, trackables, 'clean_total_1')
        assert_no_grads(model)

        # trackables = update_trackables(model, optimizer, loss_fn, preds[preds.shape[0]//2:], targets[preds.shape[0]//2:], ratio/2, trackables, 'clean_total_2')
        # assert_no_grads(model)

        ## get nr grads 
        # trackables = update_trackables(model, optimizer, crossentropy_nr, preds, targets, ratio, trackables, 'clean_nr')
        # assert_no_grads(model)

        ## get dr grads
        # trackables = update_trackables(model, optimizer, crossentropy_dr, preds, targets, ratio, trackables, 'clean_dr')
        # assert_no_grads(model)
        

        # Noisy Examples
        preds, targets, ratio = out[clean_ids==0], labs[clean_ids==0], noisy_ratio

        ## get full grads
        trackables = update_trackables(model, optimizer, loss_fn, preds, targets, ratio, trackables, 'noisy_total_1')
        assert_no_grads(model)
        # trackables = update_trackables(model, optimizer, loss_fn, preds[preds.shape[0]//2:], targets[preds.shape[0]//2:], ratio/2, trackables, 'noisy_total_2')
        # assert_no_grads(model)

        ## get nr grads 
        # trackables = update_trackables(model, optimizer, crossentropy_nr, preds, targets, ratio, trackables, 'noisy_nr')
        # assert_no_grads(model)

        ## get dr grads
        # trackables = update_trackables(model, optimizer, crossentropy_dr, preds, targets, ratio, trackables, 'noisy_dr')
        # assert_no_grads(model)
        
        # Now we will actually do the optimization step
        loss = loss_fn(out, labs)
        trackables['total_loss'] += loss.cpu().item()
        loss.backward()
        trackables['total_total_grads'] = update_gradients(trackables['total_total_grads'], model)
        optimizer.step()
        
        trackables['total_correct'] += (out.argmax(1) == labs).sum().cpu().item()
        trackables['clean_correct'] += (out.argmax(1) == labs)[clean_ids==1].sum().cpu().item()
        trackables['noisy_correct'] += (out.argmax(1) == labs)[clean_ids==0].sum().cpu().item()
        trackables['total_num'] += ims.shape[0]
        trackables['clean_num'] += clean_ids.sum()
        trackables['noisy_num'] += noisy_ids.sum()

        if scheduler is not None: scheduler.step()
        #get lr for ptinting later
        

    # print(clean_grads[0][0],noisy_grads[0][0], total_grads[0][0], clean_grads[0][0]+noisy_grads[0][0])
    
    acc = trackables['total_correct'] / trackables['total_num']
    clean_acc = trackables['clean_correct'] / trackables['clean_num']
    noisy_acc = trackables['noisy_correct'] / trackables['noisy_num']
    clean_loss = trackables['clean_total_1_loss'] / len(loader)
    noisy_loss = trackables['noisy_total_1_loss'] / len(loader)
    # clean_loss_nr = trackables['clean_nr_loss'] / len(loader)
    # noisy_loss_nr = trackables['noisy_nr_loss'] / len(loader)
    # clean_loss_dr = trackables['clean_dr_loss'] / len(loader)
    # noisy_loss_dr = trackables['noisy_dr_loss'] / len(loader)
    loss = trackables['total_loss'] / len(loader)
    lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch + 1} \t Accuracy= {acc*100:.2f}  \t Clean = {clean_acc*100:.2f} \t Noisy = {noisy_acc*100:.2f} \
            \n \t\t Loss = {loss:.4f} \t Clean = {clean_loss:.4f} \t Noisy = {noisy_loss:.4f} \t LR = {lr:.4f}")
            # \n \t\t Nr Loss Clean = {clean_loss_nr:.4f} \t Noisy = {noisy_loss_nr:.4f} \
            # \n \t\t Dr Loss Clean = {clean_loss_dr:.4f} \t Noisy = {noisy_loss_dr:.4f} 

    
    trackables = grads_to_norms(trackables)
    return trackables

def train(args, pre_dict):
    classes_mapper = {
        "cifar10":10,
        "cifar100":100,
        "svhn":10,
        "mnist":10
    }
    in_channels = 1 if args["dataset1"] == "mnist" else 3
    image_size = 28 if args["dataset1"] == "mnist" and args["augmentation"]==0 else 32

    model = get_model(args["model_type"], NUM_CLASSES=classes_mapper[args["dataset1"]], in_channels=in_channels, image_size=image_size)

    train_loader = pre_dict["train_loader"]
    test_loader = pre_dict["test_loader"]
    optimizer = SGD(model.parameters(), lr=args["lr1"], momentum=0.9, weight_decay=5e-4)

    scheduler, EPOCHS = get_scheduler_epochs("cosine", optimizer, train_loader, max_epochs = args["num_epochs"])

    loader = train_loader
    noise_mask = pre_dict["noise_mask"]

    trackables_all_epochs = {}
    trackables_all_epochs["noise_mask"] = noise_mask
    all_models = [copy.deepcopy(model.state_dict())]

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0)

    for ep in range(EPOCHS):
        trackables = single_epoch(model, optimizer, loader, loss_fn, scheduler, noise_mask = noise_mask, track_gradients=True, epoch=ep)
        eval_rets = eval(model, test_loader, eval_mode=True)
        trackables["eval"] = eval_rets
        
        for key in trackables:
            full_key = f'{key}_all_epochs'
            if full_key not in trackables_all_epochs:
                trackables_all_epochs[full_key] = []
            trackables_all_epochs[full_key].append(trackables[key])
        
        all_models.append(copy.deepcopy(model.state_dict()))
    
    return model, all_models, trackables_all_epochs


def trainer(all_args, filename):
    #iid setting
    print(filename)
    all_args["noise_2"] = all_args["noise_1"]
    all_args["minority_2"] = all_args["minority_1"]
    pre_dict, ft_dict = return_loaders(all_args, get_frac = False, aug=all_args["augmentation"])
    model, all_models, trackables_all_epochs = train(all_args, pre_dict)
    
    with open(f'{filename}trackables.pickle', 'wb') as handle:
        pickle.dump(trackables_all_epochs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(f'{filename}models.pickle', 'wb') as handle:
        pickle.dump(all_models, handle, protocol=pickle.HIGHEST_PROTOCOL)
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
    args["cscore"] = float(f_remaining.split("_")[12])
    return args

if __name__ == "__main__":
    # wandb.init(project='test'),

    parser = params.parse_args()
    args = parser.parse_args()
    args = params.add_config(args) if args.config_file != None else args
    args = vars(args)
    if args["from_dir_name"] is not None: args = dir_to_args(args)
    args["dataset2"] = args["dataset1"]
    if args["model_type"] == "vit": args["batch_size"] = 128
    
    filename = f'logs/{args["dataset1"]}/{args["model_type"]}_lr_{args["lr1"]}_noise_{args["noise_1"]}_{args["model_type"]}_{args["sched"]}_seed_{args["seed"]}_aug_{args["augmentation"]}_cscore_{args["cscore"]}/'
    model_pickle = f'{filename}models.pickle'
    if (os.path.exists(model_pickle)):
        print("Saved model already exists")
        exit(0)

    
    if not os.path.exists(filename):
        os.makedirs(filename)
    seed_everything(args["seed"])

    sys.stdout = MyLogger(f"{filename}/out.log", "a")
    print(args)

    trainer(args, filename)