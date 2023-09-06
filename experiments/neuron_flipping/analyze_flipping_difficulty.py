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
print (filename)


seed_everything(args["seed"])
pre_dict, ft_dict = return_loaders(args, get_frac=False, shuffle = False, aug = False)
model_name =  f"{filename}{args['model_type']}_final.pt"


try:
   assert (os.path.exists(filename))
except:
   # Train the Model
   print ("######### Training Model ###########")
   model = get_model(args["model_type"], NUM_CLASSES=10)
   model = model.cuda()
   train_loader = pre_dict["train_loader"]
   optimizer = SGD(model.parameters(), lr=args["lr1"], momentum=0.9, weight_decay=5e-4)

   scheduler, EPOCHS = get_scheduler_epochs("triangle", optimizer, train_loader, max_epochs = 25)
   loss_fn = nn.CrossEntropyLoss()
   train_rets = train(model, train_loader, optimizer, scheduler, loss_fn, EPOCHS, patience = 5, eval_every = False, eval_loader= None, save_every = None, mask = None)
   #save model
   #make directory
   os.makedirs(filename, exist_ok=True)
   torch.save(model.state_dict(), model_name)

#Load Model
print ("######### Loading Saved Model ###########")
saved_model = get_model(f"{args['model_type']}")
saved_model.load_state_dict(torch.load(model_name))
train_loader = pre_dict["train_loader"]

print (f"Initial accuracy on training set = {eval(saved_model, train_loader, eval_mode = False)['accuracy']}")


num_examples = 1000


# channel_wise = channel, weight
# objective = "zero", "step"
rets = flip_preds(saved_model, 
                            loader = pre_dict["train_loader"], 
                            example_type=args["example_type"], 
                            noise_mask= torch.from_numpy(pre_dict["noise_mask"]), 
                            rare_mask = torch.from_numpy(pre_dict["rare_mask"]) if pre_dict["rare_mask"] is not None else None, 
                            eval_post_edit=True, 
                            num_examples = num_examples, 
                            verbose = False,
                            channel_wise = args["channel_wise"],
                            gaussian_noise=args["gaussian_noise"],
                            objective = args["objective"],
                            n_EoT=args["n_EoT"])


import pickle

with open(f"{filename}{args['example_type']}_flips_{args['channel_wise']}_wise_{args['objective']}_gaussian_{args['gaussian_noise']}.pickle", "wb") as output_file:
   pickle.dump(rets, output_file)