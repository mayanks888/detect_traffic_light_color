#!/usr/bin/env python

import numpy as np
import pandas as pd

import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler


from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score


# In[2]:


np.random.seed(0)
torch.manual_seed(0)


# In[3]:


# get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('darkgrid')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("We're using =>", device)

root_dir = "/home/mayank_s/datasets/color_tl_datasets/Include_all_dataset/"
print("The data lies here =>", root_dir)


image_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

pokemon_dataset = datasets.ImageFolder(root = root_dir,
                                       transform = image_transforms["train"]
                                      )

pokemon_dataset


idx2class = {v: k for k, v in pokemon_dataset.class_to_idx.items()}

def get_class_distribution(dataset_obj):
    count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}
    
    for _, label_id in dataset_obj:
        label = idx2class[label_id]
        count_dict[label] += 1
    return count_dict


def plot_from_dict(dict_obj, plot_title, **kwargs):
    return sns.barplot(data = pd.DataFrame.from_dict([dict_obj]).melt(), x = "variable", y="value", hue="variable", **kwargs).set_title(plot_title)


# In[10]:


plt.figure(figsize=(15,8))
plot_from_dict(get_class_distribution(pokemon_dataset), plot_title="Entire Dataset (before train/val/test split)")


def create_samplers(dataset, train_percent, val_percent):
    # Create a list of indices from 0 to length of dataset.
    dataset_size = len(dataset)
    dataset_indices = list(range(dataset_size))
    
    # Shuffle the list of indices using `np.shuffle`.
    np.random.shuffle(dataset_indices)
    
    # Create the split index. We choose the split index to be 20% (0.2) of the dataset size.
    train_split_index = int(np.floor(train_percent * dataset_size))
    val_split_index = int(np.floor(val_percent * dataset_size))
    

    # Slice the lists to obtain 2 lists of indices, one for train and other for test.
    # `0-------------------------- train_idx----- val_idx ---------n`

    train_idx = dataset_indices[:train_split_index]
    val_idx = dataset_indices[train_split_index:train_split_index+val_split_index]
    test_idx = dataset_indices[train_split_index+val_split_index:]
    
    # Finally, create samplers.
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    
    return train_sampler, val_sampler, test_sampler


# In[12]:


train_sampler, val_sampler, test_sampler = create_samplers(pokemon_dataset, 0.8, 0.1)



train_loader = DataLoader(dataset=pokemon_dataset, shuffle=False, batch_size=8, sampler = train_sampler)
val_loader = DataLoader(dataset=pokemon_dataset, shuffle=False, batch_size=1, sampler = val_sampler)
test_loader = DataLoader(dataset=pokemon_dataset, shuffle=False, batch_size=1, sampler = test_sampler)



def get_class_distribution_loaders(dataloader_obj, dataset_obj):
    count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}
    
    if dataloader_obj.batch_size == 1:    
        for _,label_id in dataloader_obj:
            y_idx = label_id.item()
            y_lbl = idx2class[y_idx]
            count_dict[str(y_lbl)] += 1
    else: 
        for _,label_id in dataloader_obj:
            for idx in label_id:
                y_idx = idx.item()
                y_lbl = idx2class[y_idx]
                count_dict[str(y_lbl)] += 1
        
            
    return count_dict


#
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18,7))

plot_from_dict(get_class_distribution_loaders(train_loader, pokemon_dataset), plot_title="Train Set", ax=axes[0])
plot_from_dict(get_class_distribution_loaders(val_loader, pokemon_dataset), plot_title="Val Set", ax=axes[1])


single_batch = next(iter(train_loader))



single_batch[0].shape




print("Output label tensors: ", single_batch[1])
print("\nOutput label tensor shape: ", single_batch[1].shape)



# Selecting the first image tensor from the batch. 
single_image = single_batch[0][0]
single_image.shape


# In[20]:


plt.imshow(single_image.permute(1, 2, 0))




# We do single_batch[0] because each batch is a list 
# where the 0th index is the image tensor and 1st index is the output label.
single_batch_grid = utils.make_grid(single_batch[0], nrow=4)


# In[22]:


plt.figure(figsize = (10,10))
plt.imshow(single_batch_grid.permute(1, 2, 0))


# ## Define a CNN Architecture

# We will fine tune a pretrained `mobilenet`, takek from `torchvision`, for this task.

# In[23]:


model = models.mobilenet_v2(pretrained=True, progress=True)


# In[24]:


model.classifier[1] = nn.Linear(1280, len(idx2class))


# Now we'll initialize the model, optimizer, and loss function. 
# 
# Then we'll transfer the model to GPU. 
# 
# We're using the `nn.CrossEntropyLoss` .
# 
# We don't have to manually apply a `log_softmax` layer after our final layer because `nn.CrossEntropyLoss` does that for us.
# 
# However, we need to apply `log_softmax` for our validation and testing.

# In[30]:


model.to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# Before we start our training, let's define a function to calculate accuracy per epoch. 
# 
# This function takes `y_pred` and `y_test` as input arguments. We then apply softmax to `y_pred` and extract the class which has a higher probability.
# 
# After that, we compare the the predicted classes and the actual classes to calculate the accuracy.

# In[31]:


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc) * 100
    
    return acc


# We'll also define 2 dictionaries which will store the accuracy/epoch and loss/epoch for both train and validation sets.

# In[32]:


accuracy_stats = {
    'train': [],
    "val": []
}

loss_stats = {
    'train': [],
    "val": []
}




print("Begin training.")

# for e in tqdm(range(1, 11)):
for e in (range(1, 11)):

    # TRAINING
    
    train_epoch_loss = 0
    train_epoch_acc = 0
    
    model.train()
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        
        optimizer.zero_grad()
        
        y_train_pred = model(X_train_batch).squeeze()
                
        train_loss = criterion(y_train_pred, y_train_batch)
        train_acc = multi_acc(y_train_pred, y_train_batch)
        
        train_loss.backward()
        optimizer.step()
        
        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()
        
        
    # VALIDATION
    with torch.no_grad():
        model.eval()
        val_epoch_loss = 0
        val_epoch_acc = 0
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            
            y_val_pred = model(X_val_batch).squeeze()
            
            y_val_pred = torch.unsqueeze(y_val_pred, 0)
                                    
            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = multi_acc(y_val_pred, y_val_batch)
            
            val_epoch_loss += train_loss.item()
            val_epoch_acc += train_acc.item()

    
        
    loss_stats['train'].append(train_epoch_loss/len(train_loader))
    loss_stats['val'].append(val_epoch_loss/len(val_loader))
    accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
    accuracy_stats['val'].append(val_epoch_acc/len(val_loader))
                              
    
    print(f'Epoch {e+0:02}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')


# ## Visualize Loss and Accuracy
# 
# To plot the loss and accuracy line plots, we again create a dataframe from the `accuracy_stats` and `loss_stats` dictionaries.

# In[34]:


train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30,10))
sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')


# ## Test
# 
# After training is done, we need to test how our model fared. Note that we've used `model.eval()` before we run our testing code.
# To tell PyTorch that we do not want to perform back-propagation during inference, we use `torch.no_grad()`, just like we did it for the validation loop above.
# 
# * We start by defining a list that will hold our predictions. Then we loop through our batches using the `test_loader`. For each batch -
# * We move our input mini-batch to GPU.
# * We make the predictions using our trained model.
# * Apply log_softmax activation to the predictions and pick the index of highest probability.
# * Move the batch to the GPU from the CPU.
# * Convert the tensor to a numpy object and append it to our list.

# In[35]:


y_pred_list = []
y_true_list = []
with torch.no_grad():
    for x_batch, y_batch in tqdm(test_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        y_test_pred = model(x_batch)
        
        y_test_pred = torch.log_softmax(y_test_pred, dim=1)
        _, y_pred_tag = torch.max(y_test_pred, dim = 1)
    
        y_pred_list.append(y_pred_tag.cpu().numpy())
        y_true_list.append(y_batch.cpu().numpy())


# We'll flatten out the list so that we can use it as an input to `confusion_matrix` and `classification_report`.

# In[48]:


y_pred_list = [i[0] for i in y_pred_list]
y_true_list = [i[0] for i in y_true_list]


# ## Classification Report

# Finally, we print out the classification report which contains the precision, recall, and the F1 score.

# In[75]:


# print(classification_report(y_true_list, y_pred_list))
print(f"Accuracy = {accuracy_score(y_true_list, y_pred_list) * 100}")
print(f"Precision = {precision_score(y_true_list, y_pred_list, average='weighted') * 100}")
print(f"Recall = {recall_score(y_true_list, y_pred_list, average='weighted') * 100}")
print(f"F1 Score = {f1_score(y_true_list, y_pred_list, average='weighted') * 100}")


# ## Confusion Matrix

# We create a dataframe from the confusion matrix and plot it as a heatmap using the seaborn library.

# In[53]:


confusion_matrix_df = pd.DataFrame(confusion_matrix(y_true_list, y_pred_list)).rename(columns=idx2class, index=idx2class)

fig, ax = plt.subplots(figsize=(20,20))         
sns.heatmap(confusion_matrix_df, annot=True, ax=ax)


# ## Inference

# In[142]:


test_loader_iter = iter(test_loader)


# In[151]:


single_img, single_lbl = next(test_loader_iter)
single_img, single_lbl = single_img.to(device), single_lbl.to(device)

pred = torch.log_softmax(model(single_img), dim = 1)
_, pred_class = torch.max(pred, dim = 1)
pred_class = pred_class.item()

single_img = single_img.squeeze().permute(1, 2, 0).cpu().numpy()
print(f"True Class = {idx2class[single_lbl.item()]}")
print(f"Pred Class = {idx2class[pred_class]}")

plt.imshow(single_img)

