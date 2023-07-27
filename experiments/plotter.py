from matplotlib import pyplot as plt
import numpy as np
import os, sys
import pickle
#import cm package for plotting virdis
import matplotlib.cm as cm
from layer_groups import *
import torch

def plot_accuracy_curves(directory_name):
    
    accuracy_files = ["clean_acc.npy", "noisy_acc.npy", "total_acc.npy"]
    colors = ["green", "red", "blue"]
    labels = ["Clean", "Noisy", "Total"]
    # markers = ["o", "x", "s"]
    plt.figure(figsize=(4.5,3))
    for i in range(len(accuracy_files)):
        # we need to average out the accuracies for every 3 seeds. so we will have to get the other files fro, two different directories
        #the first directory is the one we are in
        #sample dir name localizing_memorization/logs/mnist/resnet9_lr_0.001_noise_0.1_resnet9_cosine_seed_4_aug_0
        #now change the number after seed to seed+1 and seed +2 to get the other directories. 
        # index of seed in the directory name
        seed_index = directory_name.rfind("seed_") + 5
        current_seed = int(directory_name[seed_index])
        dir_2 = directory_name[:seed_index] + str(current_seed + 1) + directory_name[seed_index + 1:]
        dir_3 = directory_name[:seed_index] + str(current_seed + 2) + directory_name[seed_index + 1:]

        #if any of the files does not exist, take an average over the remaining
        accs = []
        acc_1 = np.load(os.path.join(directory_name, accuracy_files[i]))
        accs.append(acc_1)
        if os.path.exists(os.path.join(dir_2, accuracy_files[i])) :
            acc_2 = np.load(os.path.join(dir_2, accuracy_files[i])) 
            accs.append(acc_2)

        if os.path.exists(os.path.join(dir_3, accuracy_files[i])) :
            acc_3 = np.load(os.path.join(dir_3, accuracy_files[i])) 
            accs.append(acc_3)

        acc_mean = np.mean(accs, axis=0)
        acc_std = np.std(accs, axis=0)
        plt.plot(range(len(acc_mean)), acc_mean, color=colors[i], label=labels[i])#, marker=markers[i])
        plt.fill_between(range(len(acc_mean)), acc_mean - acc_std, acc_mean + acc_std, color=colors[i], alpha=0.2)

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    # add gridlines on both axes
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha = 0.2)
    # plt.minorticks_on()
    # plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)


    plt.savefig(os.path.join(directory_name, "plots/accuracy_curves.pdf"), bbox_inches='tight')

def plot_loss_curves(directory_name):
    loss_files = ["clean_loss.npy", "noisy_loss.npy", "total_loss.npy"]
    colors = ["green", "red", "blue"]
    labels = ["Clean", "Noisy", "Total"]
    plt.figure()
    for i in range(len(loss_files)):
        #like the previous function, adapt this to get mean variance
        seed_index = directory_name.rfind("seed_") + 5
        current_seed = int(directory_name[seed_index])
        dir_2 = directory_name[:seed_index] + str(current_seed + 1) + directory_name[seed_index + 1:]
        dir_3 = directory_name[:seed_index] + str(current_seed + 2) + directory_name[seed_index + 1:]
        losses = []
        loss_1 = np.load(os.path.join(directory_name, loss_files[i]))
        losses.append(loss_1)
        if os.path.exists(os.path.join(dir_2, loss_files[i])) :
            loss_2 = np.load(os.path.join(dir_2, loss_files[i])) 
            losses.append(loss_2)
        if os.path.exists(os.path.join(dir_3, loss_files[i])) :
            loss_3 = np.load(os.path.join(dir_3, loss_files[i])) 
            losses.append(loss_3)
        
        loss_mean = np.mean(losses, axis=0)
        loss_std = np.std(losses, axis=0)
        plt.plot(range(len(loss_mean)), loss_mean, color=colors[i], label=labels[i])
        plt.fill_between(range(len(loss_mean)), loss_mean - loss_std, loss_mean + loss_std, color=colors[i], alpha=0.2)


    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # add gridlines on both axes
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha = 0.2)
    # plt.minorticks_on()
    # plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    plt.savefig(os.path.join(directory_name, "plots/loss_curves.pdf"), bbox_inches='tight')

def plot_dr_loss_curves(directory_name):
    loss_files_dr = ["clean_loss_dr.npy", "noisy_loss_dr.npy"]#, "total_loss_dr.npy"]
    loss_files_nr = ["clean_loss_nr.npy", "noisy_loss_nr.npy"]#, "total_loss_nr.npy"]

    #dr loss in dotted, and nr loss in solid lines

    colors = ["green", "red", "blue"]
    labels = ["Clean", "Noisy", "Total"]
    plt.figure()
    
    for i in range(len(loss_files_dr)):
        #like the previous function, adapt this to get mean variance
        seed_index = directory_name.rfind("seed_") + 5
        current_seed = int(directory_name[seed_index])
        dir_2 = directory_name[:seed_index] + str(current_seed + 1) + directory_name[seed_index + 1:]
        dir_3 = directory_name[:seed_index] + str(current_seed + 2) + directory_name[seed_index + 1:]
        losses_dr = []
        losses_nr = []
        loss_1_dr = np.load(os.path.join(directory_name, loss_files_dr[i]))
        loss_1_nr = np.load(os.path.join(directory_name, loss_files_nr[i]))
        losses_dr.append(loss_1_dr)
        losses_nr.append(loss_1_nr)
        if os.path.exists(os.path.join(dir_2, loss_files_dr[i])) :
            loss_2_dr = np.load(os.path.join(dir_2, loss_files_dr[i])) 
            loss_2_nr = np.load(os.path.join(dir_2, loss_files_nr[i])) 
            losses_dr.append(loss_2_dr)
            losses_nr.append(loss_2_nr)
        if os.path.exists(os.path.join(dir_3, loss_files_dr[i])) :
            loss_3_dr = np.load(os.path.join(dir_3, loss_files_dr[i])) 
            loss_3_nr = np.load(os.path.join(dir_3, loss_files_nr[i])) 
            losses_dr.append(loss_3_dr)
            losses_nr.append(loss_3_nr)

        loss_mean_dr = np.mean(losses_dr, axis=0)
        loss_std_dr = np.std(losses_dr, axis=0)
        loss_mean_nr = np.mean(losses_nr, axis=0)
        loss_std_nr = np.std(losses_nr, axis=0)
        plt.plot(range(len(loss_mean_dr)), loss_mean_dr, color=colors[i], label=labels[i], linestyle="dotted")
        plt.fill_between(range(len(loss_mean_dr)), loss_mean_dr - loss_std_dr, loss_mean_dr + loss_std_dr, color=colors[i], alpha=0.2, linestyle="dotted")
        plt.plot(range(len(loss_mean_nr)), loss_mean_nr, color=colors[i], label=labels[i], linestyle="solid")
        plt.fill_between(range(len(loss_mean_nr)), loss_mean_nr - loss_std_nr, loss_mean_nr + loss_std_nr, color=colors[i], alpha=0.2, linestyle="solid")


    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # add gridlines on both axes
    # plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha = 0.2)
    # plt.minorticks_on() 
    # plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    plt.savefig(os.path.join(directory_name, "plots/loss_curves_dr_nr.pdf"), bbox_inches='tight')


from gradients import get_all_grads, get_all_grads_cosine_similarity
def plot_grads(dir_name):

    #get directory name from directory
    plots = os.path.join(dir_name, "plots")
    if not os.path.exists(plots):
        os.mkdir(plots)

    # we need to get gradients for all the three directories
    #find names of other two directories
    seed_index = dir_name.rfind("seed_") + 5
    current_seed = int(dir_name[seed_index])
    dir_2 = dir_name[:seed_index] + str(current_seed + 1) + dir_name[seed_index + 1:]
    dir_3 = dir_name[:seed_index] + str(current_seed + 2) + dir_name[seed_index + 1:]

    gt, gc, gn = [], [], []
    for dir in [dir_name, dir_2, dir_3]:
        trackables_filename = os.path.join(dir, "trackables.pickle")
        if os.path.exists(trackables_filename):
            grads_total, grads_clean, grads_noisy = get_all_grads(trackables_filename)
            gt.append(grads_total)
            gc.append(grads_clean)
            gn.append(grads_noisy)


    #get the mean of the three directories
    grads_total = np.mean(gt, axis=0)
    grads_clean = np.mean(gc, axis=0)
    grads_noisy = np.mean(gn, axis=0)

    #get the std of the three directories
    std_total = np.std(gt, axis=0)
    std_clean = np.std(gc, axis=0)
    std_noisy = np.std(gn, axis=0)


    num_epochs = grads_total.shape[0]
    num_groups = grads_total.shape[1]
    
    #the x axis should represent a group

    # we will draw a different plot for the average of every 5 epochs. y axis is the average norm. clean, noisy and total are shown with green, red and blue colors respectively

    for i in range(0, num_epochs, 5):
        #get the mean of the next 5 epochs
        mean_total = np.mean(grads_total[i:i+5], axis=0)
        mean_clean = np.mean(grads_clean[i:i+5], axis=0)
        mean_noisy = np.mean(grads_noisy[i:i+5], axis=0)

        #also take the mean of std
        mean_std_total = np.mean(std_total[i:i+5], axis=0)
        mean_std_clean = np.mean(std_clean[i:i+5], axis=0)
        mean_std_noisy = np.mean(std_noisy[i:i+5], axis=0)

        
        #create a new figure
        plt.figure(figsize=(4,3))
        plt.plot(range(num_groups), mean_total, color='blue')
        plt.plot(range(num_groups), mean_clean, color='green')
        plt.plot(range(num_groups), mean_noisy, color='red')

        #fill with std
        plt.fill_between(range(num_groups), mean_total - mean_std_total, mean_total + mean_std_total, color='blue', alpha=0.2)
        plt.fill_between(range(num_groups), mean_clean - mean_std_clean, mean_clean + mean_std_clean, color='green', alpha=0.2)
        plt.fill_between(range(num_groups), mean_noisy - mean_std_noisy, mean_noisy + mean_std_noisy, color='red', alpha=0.2)

        # add legend    
        plt.legend(['Total', 'Clean', 'Noisy'])
        plt.xlabel('Model Layer Depth')
        plt.ylabel('Average Gradient Norm')
        # plt.title('average norm of gradients for epochs {} to {}'.format(i, i+5))
        #save figures by making a new sub folder in the same directory that contains trackables.pickle
        plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)
        # plt.minorticks_on() 
        # plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.savefig(f'{plots}/grads_{i}_{i+5}.pdf', bbox_inches='tight')



def plot_rewinding_curves(directory_name):
    
    #average over 3 seeds
    #find names of other two directories
    seed_index = directory_name.rfind("seed_") + 5
    current_seed = int(directory_name[seed_index])
    dir_2 = directory_name[:seed_index] + str(current_seed + 1) + directory_name[seed_index + 1:]
    dir_3 = directory_name[:seed_index] + str(current_seed + 2) + directory_name[seed_index + 1:]


    #load the rewinding pickle
    c_a = []
    n_a = []

    dir_1 = directory_name
    for dir in [dir_1, dir_2, dir_3]:
        rwf = os.path.join(dir, "rewinding.pickle")
        if os.path.exists(rwf):
            with open(rwf,'rb') as f:
                rewinding_pickle  = pickle.load(f)
            trk_fname = os.path.join(dir, "trackables.pickle")
            with open(trk_fname, 'rb') as f:
                trk = pickle.load(f)
                noise_mask = trk["noise_mask"]
   
            #final shape of outputs will be num_epochs x num_groups
            noisy_accs = np.zeros((len(rewinding_pickle), len(rewinding_pickle[0])))
            clean_accs = np.zeros((len(rewinding_pickle), len(rewinding_pickle[0])))

            for i in range(len(rewinding_pickle)):
                for j in range(len(rewinding_pickle[0])):
                    noisy_accs[i][j] = (np.array(rewinding_pickle[i][j]["acc_mask"]) == 1)[noise_mask==1].mean()
                    clean_accs[i][j] = (np.array(rewinding_pickle[i][j]["acc_mask"]) == 1)[noise_mask==0].mean()
            c_a.append(clean_accs)
            n_a.append(noisy_accs)



    #get the mean of the three directories
    rewinding_clean = np.mean(c_a, axis=0)
    rewinding_noisy = np.mean(n_a, axis=0)
    #std
    rewinding_std_clean = np.std(c_a, axis=0)
    rewinding_std_noisy = np.std(n_a, axis=0)

    #rewinding is shape num_epochs x num_groups
    #we will plot a different line for each epoch [0,10,20,30...]
    #the colour of the lines should keep getting darker
    #the x axis should represent a group
    #the y axis should represent the accuracy
    
    #only keep first 6 lines
    rewinding_clean = rewinding_clean[:6]
    num_lines = rewinding_clean.shape[0]
    num_groups = rewinding_clean.shape[1]
    


    for ex_type in ["clean", "noisy"]:
        rewinding = rewinding_clean if ex_type == "clean" else rewinding_noisy
        rewinding_std = rewinding_std_clean if ex_type == "clean" else rewinding_std_noisy
        plt.figure(figsize=(4,3))

        #make a custom color bar of colors of green 
        # colorbar = np.zeros((num_lines, 3))
        # for i in range(num_lines):
        #     colorbar[i] = (0, 1.0 - i/num_lines, 0) if ex_type == "clean" else (1.0 - i/num_lines, 0, 0)
        # plt.imshow(colorbar, aspect='auto', extent=[0, num_groups, 0, num_lines])

        for i in range(0, num_lines):
            #get the mean of the next 10 epochs
            mean = rewinding[i]
            #also take the mean of std
            std = rewinding_std[i]
            #create a new figure using different shades of green colour. i do not want to use virdis colour palette.
            c =(0, 1.0 - i/num_lines, 0) if ex_type == "clean" else (1.0 - i/num_lines, 0, 0)

            #add circle markers for clean, and star markers for noisy 
            plt.plot(range(num_groups), mean, color = c, label = i*10, marker = "o" if ex_type == "clean" else "*")
            #fill with std using different shades of green colour
            plt.fill_between(range(num_groups), mean - std, mean + std, color = c, alpha=0.2)

        #give heading "Epoch" to legend box and place it to the right of the figure outside it
        plt.legend(title = "Epoch", loc = 'center left', bbox_to_anchor = (1, 0.5))



        plt.xlabel('Model Layer Depth')
        plt.ylabel('Accuracy')
        # plt.title('Accuracy after rewinding to different epochs')
        #save figures by making a new sub folder in the same directory that contains trackables.pickle
        plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)
        # plt.minorticks_on()
        # plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plots = os.path.join(directory_name, "plots")
        plt.savefig(f'{plots}/rewinding_{ex_type}.pdf', bbox_inches='tight')




def plot_pearson_correlation(directory_name):
    #load all model weights
    #load all trackables
    #for each model, get the pearson correlation between the change in weights over an epoch and the change in loss on noisy and clean examples for that epoch

    #average over 3 seeds
    #find names of other two directories
    seed_index = directory_name.rfind("seed_") + 5
    current_seed = int(directory_name[seed_index])
    dir_2 = directory_name[:seed_index] + str(current_seed + 1) + directory_name[seed_index + 1:]
    dir_3 = directory_name[:seed_index] + str(current_seed + 2) + directory_name[seed_index + 1:]
    loss_files = ["clean_loss.npy", "noisy_loss.npy"]

    model_type = "resnet50" if "resnet50" in directory_name else "vit" if "vit" in directory_name else "resnet9"
    weight_groups = {"vit":vit_groups, "resnet50":resnet50_groups,"resnet9":resnet9_groups}[model_type]


    all_losses = []
    all_weight_diff = []
    for dir in [directory_name, dir_2, dir_3]:
        if os.path.exists(os.path.join(dir, "trackables.pickle")):
            trk_fname = os.path.join(dir, "trackables.pickle")
            with open(trk_fname, 'rb') as f:
                trk = pickle.load(f)
                noise_mask = trk["noise_mask"]
        
            model_name = os.path.join(dir, "models.pickle")
            with open(model_name, 'rb') as f:
                models = pickle.load(f)
            
            #load loss files
            loss_diffs = []
            for loss_file in loss_files:
                loss_name = os.path.join(dir, loss_file)
                #load numpy
                loss = np.load(loss_name)
                #get change in loss per epoch (subtract each element from the previous element)
                loss_change = np.diff(loss)
                loss_diffs.append(loss_change)
            all_losses.append(loss_diffs)

            weight_diffs = []
            for i in range(len(models) - 1):
                model_1 = models[i]
                model_2 = models[i + 1]
                #get the difference in weights for each layer
                weight_diff = {}
                for layer_name in model_1.keys():
                    weight_1 = model_1[layer_name]
                    weight_2 = model_2[layer_name]
                    weight_diff_layer = weight_2 - weight_1
                    #normalize by the sqrt of size of the layer
                    weight_diff_layer = weight_diff_layer / np.sqrt(weight_diff_layer.numel())
                    weight_diff[layer_name] = weight_diff_layer
                    
                weight_diffs.append(weight_diff)

            #now we have a list of weight_diffs for each epoch
            #we will now average over all layers in a group
            weight_diff_group_all_epochs = []
            for epoch in range(len(weight_diffs)):
                weight_diff = weight_diffs[epoch]
                weight_diff_group = {}
                for group_name in weight_groups:
                    weight_diff_group[group_name] = 0
                    for layer_name in weight_groups[group_name]:
                        weight_diff_group[group_name] += (weight_diff[layer_name]).norm().item()
                    weight_diff_group[group_name] = weight_diff_group[group_name] / len(weight_groups[group_name])
                weight_diff_group_all_epochs.append(weight_diff_group)

            weight_diffs_per_epoch = []
            for group_name in weight_diff_group_all_epochs[0]:
                weight_diffs_per_epoch.append([weight_diff_group_all_epochs[i][group_name] for i in range(len(weight_diff_group_all_epochs))])

            all_weight_diff.append(weight_diffs_per_epoch)
       
        #average over 3 seeds
    weight_diffs_per_epoch = np.mean(all_weight_diff, axis=0)
    loss_diffs = np.mean(all_losses, axis=0)
    
    
    from scipy.stats import pearsonr

    #clean pearson correlation
    pearson_correlations_clean = []
    for i in range(len(weight_diffs_per_epoch)):
        pearson_correlations_clean.append(pearsonr(weight_diffs_per_epoch[i][10:30], loss_diffs[0][10:30])[0])
    
    #noisy pearson correlation
    pearson_correlations_noisy = []
    for i in range(len(weight_diffs_per_epoch)):
        pearson_correlations_noisy.append(pearsonr(weight_diffs_per_epoch[i][10:30], loss_diffs[1][10:30])[0])

    #plot the pearson correlation for each group
    plt.figure()
    plt.plot(pearson_correlations_clean, label="clean")
    plt.plot(pearson_correlations_noisy, label="noisy")
    plt.legend()
    plt.xlabel("Layer Number")
    plt.ylabel("Pearson Correlation")
    plt.title("Pearson Correlation between Weight Change and Loss Change")
    plots = os.path.join(directory_name, "plots")
    plt.savefig(os.path.join(plots, "pearson_correlation.pdf"), bbox_inches='tight')


def plot_retraining_curves(directory_name):
    #load all model weights
    #load all trackables
    #for each model, get the pearson correlation between the change in weights over an epoch and the change in loss on noisy and clean examples for that epoch

    #average over 3 seeds
    #find names of other two directories
    seed_index = directory_name.rfind("seed_") + 5
    current_seed = int(directory_name[seed_index])
    dir_2 = directory_name[:seed_index] + str(current_seed + 1) + directory_name[seed_index + 1:]
    dir_3 = directory_name[:seed_index] + str(current_seed + 2) + directory_name[seed_index + 1:]

    model_type = "resnet50" if "resnet50" in directory_name else "vit" if "vit" in directory_name else "resnet9"
    weight_groups = {"vit":vit_groups, "resnet50":resnet50_groups,"resnet9":resnet9_groups}[model_type]

    # retraining_files = []
    for dir in [directory_name, dir_2, dir_3]:
        if os.path.exists(os.path.join(dir, "reatraining.pickle")):
            retraining_file_name = os.path.join(dir, "reatraining.pickle")
            #load pickle
            with open(retraining_file_name, 'rb') as f:
                retraining_file = pickle.load(f)["clean_training"]
                # retraining_files.append(retraining_file)
        break 
        

    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(3, 3, figsize=(4,4))
    # fig.autofmt_xdate()

    i,j,z = 0,0,0
    colors = ["red", "green"]

    #show the y label of Accuracy only once for the entire grid of 9 plots
    #show the x label of Epochs only once for the entire grid of 9 plots
    axs[1,0].set_ylabel("Accuracy", fontsize=14)
    axs[2,1].set_xlabel("Epochs", fontsize=14)

    for key in retraining_file.keys():
        clean_accs, noisy_accs = retraining_file[key]
        x_axis = np.arange(len(clean_accs))

        #show y ticks
        axs[i,j].set_yticks([0.5,1.0])
        axs[i,j].set_yticklabels(["0.5", "1.0"], fontsize=10)
        axs[i,j].set_xticks([0, len(clean_accs)-1])
        axs[i,j].set_xticklabels(["0", str(len(clean_accs)-1)], fontsize=10)

        # axs[i,j].set_xlabel("Epochs")
        axs[i,j].plot(x_axis, torch.tensor(clean_accs).cpu(), color=colors[1], marker="o", markersize=5)
        axs[i,j].plot(x_axis, torch.tensor(noisy_accs).cpu(), color=colors[0], marker = "x",markersize=5)
        axs[i,j].set_title(f"Layer {key}", fontsize=14)
        axs[i,j].set_ylim(0.1,1.01)
        #add x and y labels
#

        j = (j+1)%3
        z = (z+1)
        i = z//3
    
    fig.tight_layout()
    legend_labels = ["Clean", "Noisy"]
    # add legend at the top of the plot in a single row. add a top padding between the legend and the plot
    fig.legend(legend_labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.15), bbox_transform=fig.transFigure, fontsize=14)

    plots = os.path.join(directory_name, "plots")
    plt.savefig(f'{plots}/retraining.pdf', bbox_inches='tight')


def plot_gradient_similarity(directory_name):
    # this funcion plots the similarity between the gradients of the clean and noisy examples
    #get directory name from directory
    plots = os.path.join(directory_name, "plots")
    if not os.path.exists(plots):
        os.mkdir(plots)

    # we need to get gradients for all the three directories
    #find names of other two directories
    seed_index = directory_name.rfind("seed_") + 5
    current_seed = int(directory_name[seed_index])
    dir_2 = directory_name[:seed_index] + str(current_seed + 1) + directory_name[seed_index + 1:]
    dir_3 = directory_name[:seed_index] + str(current_seed + 2) + directory_name[seed_index + 1:]

    gt = []
    for dir in [directory_name, dir_2, dir_3]:
        trackables_filename = os.path.join(dir, "trackables.pickle")
        if os.path.exists(trackables_filename):
            grads_cosine_similarities = get_all_grads_cosine_similarity(trackables_filename)
            gt.append(grads_cosine_similarities)


    #get the mean of the three directories
    grads_total = np.mean(gt, axis=0)

    #create a 2d array of the cosine similarities
    #each row is an epoch
    #each column is a group

    import seaborn as sns
    plt.figure(figsize=(4,3))

    ax = sns.heatmap(grads_total[:30], linewidth=0.5)

    plt.xlabel("Layer Number")
    plt.ylabel("Epoch Number")

    # ax.set_xticks(l)
    layer_numbers = [str(i) for i in range(len(grads_total[0]))]
    ax.set_xticklabels(layer_numbers)
    #make y ticks appear vertically
    ax.tick_params(axis='y', rotation=0)
    plt.show()
    plots = os.path.join(directory_name, "plots")
    plt.savefig(f"{plots}/grads_cosine_similarities.pdf", bbox_inches='tight')



if __name__ == "__main__":
    directory_name = sys.argv[1]
    if not os.path.exists(os.path.join(directory_name, "plots")):
        os.mkdir(os.path.join(directory_name, "plots"))

    #set font to latex style and increase font size by 2
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.size': 14})

    # plot_loss_curves(directory_name)
    # plot_dr_loss_curves(directory_name)
    # plot_accuracy_curves(directory_name)
    plot_grads(directory_name)
    # plot_rewinding_curves(directory_name)
    # plot_retraining_curves(directory_name)
    # plot_pearson_correlation(directory_name)
    plot_gradient_similarity(directory_name)