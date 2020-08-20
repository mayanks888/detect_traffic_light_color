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
normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


# def test(args, model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()
#
#     test_loss /= len(test_loader.dataset)
#
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))
#


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

def main():
    # Training settings
    # training_dir = "/home/mayank_sati/Desktop/traffic_light/sorting_light/final_sort"
    training_dir = "/home/mayank_s/datasets/color_tl_datasets/Include_all_dataset"
    parser = argparse.ArgumentParser(description='PyTorch  Example')
    parser.add_argument('--train_dir', type=str, default=training_dir, metavar='N',
                        help='path for training directory')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
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

    res = 32

    model=Color_Net_CNN()
    model.to(device)


    natural_img_dataset= dset.ImageFolder(root=args.train_dir)
    train_dataset = Create_Image_Datasets(imageFolderDataset=natural_img_dataset, transform=transforms.Compose(
        [transforms.Resize((res, res)), transforms.ToTensor(), normalize]), should_invert=False)

    dataset_size = len(natural_img_dataset)
    dataset_indices = list(range(dataset_size))
    np.random.shuffle(dataset_indices)
    val_split_index = int(np.floor(0.2 * dataset_size))
    train_idx, val_idx = dataset_indices[val_split_index:], dataset_indices[:val_split_index]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(dataset=train_sampler, shuffle=False, batch_size=32,num_workers=4, drop_last=True)
    val_loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=1, sampler=val_sampler)
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))
    # sns.barplot(data=pd.DataFrame.from_dict([get_class_distribution_loaders(train_loader, natural_img_dataset)]).melt(),
    #             x="variable", y="value", hue="variable", ax=axes[0]).set_title('Train Set')
    # sns.barplot(data=pd.DataFrame.from_dict([get_class_distribution_loaders(val_loader, natural_img_dataset)]).melt(),
    #             x="variable", y="value", hue="variable", ax=axes[1]).set_title('Val Set')
    #



    # train_folder_dataset, val_folder_dataset = random_split(folder_dataset, (1500-400, 113+400))
    # train_loader = DataLoader(dataset=train_folder_dataset, shuffle=True, batch_size=1)
    # val_loader = DataLoader(dataset=val_folder_dataset, shuffle=False, batch_size=1)
    # print("Length of the train_loader:", len(train_loader))
    # print("Length of the val_loader:", len(val_loader))
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    1
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    #
    #
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        # test(args, model, device, test_loader)

    # if (args.save_model):
    #     torch.save(model.state_dict(), "color_model.pt")
    #     # torch.save(model.state_dict(), './model-save_dict_color-%s.pt' % epoch)
    #

if __name__ == '__main__':
    main()