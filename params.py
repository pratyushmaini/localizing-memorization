import argparse
from distutils import util
import yaml
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Forgetting Time')
    
    ## Basics
    parser.add_argument("--config_file", help="Configuration file containing parameters", type=str)
    parser.add_argument("--from_dir_name", help="Directory file name parse to get parameters", type=str, default = None)
    parser.add_argument("--debug_mode", help="Switch Off Logging", action='store_true', default=False)
    parser.add_argument("--augmentation", help="Switch Off Train Augh", type=int, default=0)
    parser.add_argument("--shuffle", help="Train shuffle", type=int, default=1)
    parser.add_argument("--model_type", help="Model Architecture", type=str, 
                        default = "lenet", 
                        choices = ["resnet50","resnet9","lenet", "resnet18",
                                    "vgg11", "vgg16","vit",
                                    "resnet9_drop",
                                    "resnet18_drop",
                                    "resnet50_drop",])

    parser.add_argument("--dataset1", help="Select dataset for first split", type=str, default = "mnist", choices = ["mnist","fashionmnist","emnist","cifar10","mnist_cifar_union","cifar-5m","cifar100","cifar10_500k","imagenette","cifar10_dcgan", "svhn"])
    parser.add_argument("--dataset2", help="Select dataset for second split", type=str, default = None, choices = ["mnist","fashionmnist","emnist","cifar10","mnist_cifar_union","cifar-5m","cifar100","cifar10_500k","imagenette","cifar10_dcgan", "svhn"])
    parser.add_argument("--id", help="identifier", type=str, default = '0')

    ## Add Noise
    parser.add_argument("--noise_1", help="Fraction of Label Noise in Dataset 1", type=float, default = 0)
    parser.add_argument("--noise_2", help="Fraction of Label Noise in Dataset 1", type=float, default = 0)

    parser.add_argument("--minority_1", help="Fraction of Rare Minority Group in Dataset 1", type=float, default = 0)
    parser.add_argument("--minority_2", help="Fraction of Rare Minority Group in Dataset 1", type=float, default = 0)
    parser.add_argument("--log_factor", help="For singleton in CIFAR100 superclass", type=int, default = 0)

    parser.add_argument("--batch_size", help = "Batch Size for Train Set (Default = 512)", type = int, default = 512)
    parser.add_argument("--model_id", help = "For Saving", type = str, default = '0')
    parser.add_argument("--seed", help = "Seed", type = int, default = 0)
    parser.add_argument("--model_seed", help = "Seed", type = int, default = 0)
    parser.add_argument("--seed_superclass", help = "Seed for CIFAR100 superclass", type = int, default = 0)
    parser.add_argument("--num_epochs", help = "Number of Epochs", type = int, default = 100)
    
    #HPARAMS
    parser.add_argument("--sched", help = "triangle/step", type = str, default = "triangle")
    parser.add_argument("--opt", help = "Optimizer", type = str, default = "SGD")
    parser.add_argument("--lr1", help = "Max LR for first training", type = float, default = 0.1)
    parser.add_argument("--lr2", help = "Max LR for second training", type = float, default = 0.1)
    parser.add_argument("--wd", help = "Weight Decay", type = float, default = 5e-4)

    #TEST
    parser.add_argument("--path", help = "Path for test model load", type = str, default = None)
    parser.add_argument("--removal_metric", help = "For removing examples", type = str, default = None)


    #ATTRIBUTION ANALYSIS
    parser.add_argument("--example_type", help = "noisy/rare/clean", type = str, default = None, choices=["noisy", "rare", "clean"])
    parser.add_argument("--channel_wise", help = "For removing examples", type = str, default = None, choices=["channel", "weight"])
    parser.add_argument("--objective", help = "For removing examples", type = str, default = None, choices=["zero", "step"])
    parser.add_argument("--n_EoT", help = "For removing examples", type = int, default = 5)
    parser.add_argument("--gaussian_noise", help = "For removing examples", type = int, default = 1)


    # Dropout
    parser.add_argument("--p_fixed", help = "For removing examples", type = float, default = 0)
    parser.add_argument("--p_mem", help = "For removing examples", type = float, default = 0)
    parser.add_argument("--fac", help = "For removing examples", type = int, default = 1)
    parser.add_argument("--drop_type", help = "For removing examples", type = str, default = "example_tied")
    
    
    parser.add_argument("--cscore", help = "For removing examples", type = float, default = 0)
    
    return parser

def add_config(args):
    data = yaml.load(open(args.config_file,'r'))
    args_dict = args.__dict__
    for key, value in data.items():
        if('--'+key in sys.argv and args_dict[key] != None): ## Giving higher priority to arguments passed in cli
            continue
        if isinstance(value, list):
            args_dict[key] = []
            args_dict[key].extend(value)
        else:
            args_dict[key] = value
    return args