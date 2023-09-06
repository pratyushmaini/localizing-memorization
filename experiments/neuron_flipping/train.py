import sys

sys.path.append("../")
from utils import *
from models import *
from dataloader import *
import argparse
from attribution_utils import *

import params

parser = params.parse_args()
args = parser.parse_args()
args = params.add_config(args) if args.config_file != None else args
args = vars(args)
args["dataset2"] = args["dataset1"]
args["noise_2"] = args["noise_1"]
print(args)
filename = f'logs/{args["dataset1"]}/{args["model_type"]}_lr_{args["lr1"]}_noise_{args["noise_1"]}_{args["model_type"]}_{args["sched"]}_seed_{args["seed"]}_model-seed{args["model_seed"]}/'
if not os.path.exists(filename):
   os.makedirs(filename)
seed_everything(args["seed"])


pre_dict, ft_dict = return_loaders(args, get_frac=False)

model_name =  f"{filename}{args['model_type']}_final.pt"

## Train the model
train_loader = pre_dict["train_loader"]
#set model seed
torch.manual_seed(args["model_seed"])
model = get_model(f"{args['model_type']}")
optimizer = SGD(model.parameters(), lr=args["lr1"], momentum=0.9, weight_decay=args["wd"])

scheduler, EPOCHS = get_scheduler_epochs(args["sched"], optimizer, train_loader, max_epochs = args["num_epochs"])
loss_fn = nn.CrossEntropyLoss()
ret_pre = train(model, train_loader, optimizer, scheduler, loss_fn, EPOCHS = EPOCHS,
    eval_every = False, save_every = None, eval_loader= pre_dict["test_loader"])

## Save Model 
torch.save(model.state_dict(), model_name)

#Load Model
print ("######### Loading Saved Model ###########")
saved_model = get_model(f"{args['model_type']}")
saved_model.load_state_dict(torch.load(model_name))
train_loader = pre_dict["train_loader"]

print (f"Initial accuracy on training set = {eval(saved_model, train_loader, eval_mode = True)['accuracy']}")