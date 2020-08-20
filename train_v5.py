from __future__ import print_function
import PIL.ImageOps
from models import  *
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler,Subset
from torchsampler import ImbalancedDatasetSampler
from sampler import BalancedBatchSampler
from custom_utils import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, utils, datasets
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score

normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
col_jitter = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_epoch_acc=0
    train_total_loss=0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_acc = multi_acc(output, target)
        train_epoch_acc += train_acc
        train_total_loss+=loss
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),train_acc))
            # train_epoch_acc += train_acc

    print('epoch accuracy acc',train_epoch_acc/len(train_loader))
    return ((train_epoch_acc/len(train_loader)),(train_total_loss/len(train_loader)))


def val(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return ((correct / len(test_loader.dataset)), (test_loss / len(test_loader.dataset)))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return ((correct / len(test_loader.dataset)), (test_loss / len(test_loader.dataset)))



class Create_Image_Datasets(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        # img0_tuple = random.choice(self.imageFolderDataset.samples)

        img0 = Image.open(img0_tuple[0])
        img0 = img0.convert("RGB")
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            # img1 = PIL.ImageOps.invert(img1)
        if self.transform is not None:
            img0 = self.transform(img0)
        # return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
        label = img0_tuple[1]
        return img0, label

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


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
    val_idx = dataset_indices[train_split_index:train_split_index + val_split_index]
    test_idx = dataset_indices[train_split_index + val_split_index:]

    # Finally, create samplers.
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    return train_sampler, val_sampler, test_sampler

def main():

    # Training settings
    # training_dir = "/home/mayank_sati/Desktop/traffic_light/sorting_light/final_sort"
    training_dir = "/home/mayank_s/datasets/color_tl_datasets/Include_all_dataset"
    parser = argparse.ArgumentParser(description='PyTorch  Example')
    parser.add_argument('--train_dir', type=str, default=training_dir, metavar='N',
                        help='path for training directory')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
########################################33

    res=32
    natural_img_folder= dset.ImageFolder(root=args.train_dir)
    natural_img_dataset = Create_Image_Datasets(imageFolderDataset=natural_img_folder, transform=transforms.Compose(
        [transforms.Resize((res, res)), transforms.ToTensor(), normalize]), should_invert=False)
    ######################################################################333
    # check datasets structure

    train_sampler, val_sampler, test_sampler = create_samplers(natural_img_dataset, 0.6, 0.4)
    train_loader = DataLoader(dataset=natural_img_dataset, shuffle=False, batch_size=8, sampler=train_sampler)
    val_loader = DataLoader(dataset=natural_img_dataset, shuffle=False, batch_size=1, sampler=val_sampler)
    test_loader = DataLoader(dataset=natural_img_dataset, shuffle=False, batch_size=1, sampler=test_sampler)

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    idx2class = {v: k for k, v in natural_img_folder.class_to_idx.items()}
    show=False
    if show:
        plt.figure(figsize=(15, 8))
        plot_from_dict(get_class_distribution(natural_img_folder, idx2class), plot_title="Entire Dataset (before train/val/test split)")

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))
        plot_from_dict(get_class_distribution_loaders(train_loader, natural_img_folder,idx2class), plot_title="Train Set", ax=axes[0])
        plot_from_dict(get_class_distribution_loaders(val_loader, natural_img_folder,idx2class), plot_title="Val Set", ax=axes[1])

    ##################################################33
    show=False
    if show:
        single_batch = next(iter(train_loader))
        single_batch[0].shape
        print("Output label tensors: ", single_batch[1])
        print("\nOutput label tensor shape: ", single_batch[1].shape)
        # Selecting the first image tensor from the batch.
        single_image = single_batch[0][0]
        single_image.shape
        plt.imshow(single_image.permute(1, 2, 0))
        # We do single_batch[0] because each batch is a list
        # where the 0th index is the image tensor and 1st index is the output label.
        single_batch_grid = utils.make_grid(single_batch[0], nrow=4)
        plt.figure(figsize=(10, 10))
        plt.imshow(single_batch_grid.permute(1, 2, 0))
    ########################################################
    1

    res = 32
    model = Color_Net_CNN(len(idx2class))
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    accuracy_stats = {'train': [], "val": []}

    loss_stats = {'train': [], "val": []}

    print("Begin training.")

    for epoch in range(1, args.epochs + 1):
        # train_result_data=train(args, model, device, train_loader, optimizer, epoch)
        # val_result_data=val(args, model, device, val_loader)

        # TRAINING

        train_epoch_loss = 0
        train_epoch_acc = 0

        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)

            optimizer.zero_grad()
            y_train_pred = model(X_train_batch)
            train_loss = F.nll_loss(y_train_pred, y_train_batch)

            # y_train_pred = model(X_train_batch).squeeze()

            # train_loss = criterion(y_train_pred, y_train_batch)
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

                y_val_pred = model(X_val_batch)
                val_loss = F.nll_loss(y_val_pred, y_val_batch)
                # y_val_pred = model(X_val_batch).squeeze()

                # y_val_pred = torch.unsqueeze(y_val_pred, 0)

                # val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch)

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

        loss_stats['train'].append(train_epoch_loss / len(train_loader))
        loss_stats['val'].append(val_epoch_loss / len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc / len(val_loader))

        print(
            f'Epoch {epoch + 0:02}: | Train Loss: {train_epoch_loss / len(train_loader):.5f} | Val Loss: {val_epoch_loss / len(val_loader):.5f} | Train Acc: {train_epoch_acc / len(train_loader):.3f}| Val Acc: {val_epoch_acc / len(val_loader):.3f}')

    # ## Visualize Loss and Accuracy
    #
    # To plot the loss and accuracy line plots, we again create a dataframe from the `accuracy_stats` and `loss_stats` dictionaries.

    # In[34]:

    train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(
        columns={"index": "epochs"})
    train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(
        columns={"index": "epochs"})

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30, 10))
    sns.lineplot(data=train_val_acc_df, x="epochs", y="value", hue="variable", ax=axes[0]).set_title(
        'Train-Val Accuracy/Epoch')
    sns.lineplot(data=train_val_loss_df, x="epochs", y="value", hue="variable", ax=axes[1]).set_title(
        'Train-Val Loss/Epoch')


    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        for x_batch, y_batch in (val_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            y_test_pred = model(x_batch)

            # y_test_pred = torch.log_softmax(y_test_pred, dim=1)
            _, y_pred_tag = torch.max(y_test_pred, dim=1)

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

    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_true_list, y_pred_list)).rename(columns=idx2class,
                                                                                          index=idx2class)

    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(confusion_matrix_df, annot=True, ax=ax)
    print(confusion_matrix_df)
    # plt.iomshow()

    # ## Inference

    # In[142]:

    test_loader_iter = iter(test_loader)

    # In[151]:

    single_img, single_lbl = next(test_loader_iter)
    single_img, single_lbl = single_img.to(device), single_lbl.to(device)

    pred = torch.log_softmax(model(single_img), dim=1)
    _, pred_class = torch.max(pred, dim=1)
    pred_class = pred_class.item()

    single_img = single_img.squeeze().permute(1, 2, 0).cpu().numpy()
    print(f"True Class = {idx2class[single_lbl.item()]}")
    print(f"Pred Class = {idx2class[pred_class]}")

    plt.imshow(single_img)


# if (args.save_model):
    #     torch.save(model.state_dict(), "color_model.pt")
    #     # torch.save(model.state_dict(), './model-save_dict_color-%s.pt' % epoch)
    #

if __name__ == '__main__':
    main()