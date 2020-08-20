import seaborn as sns
import  pandas as pd
import torch
def get_class_distribution(dataset_obj,idx2class):
    count_dict = {k: 0 for k, v in dataset_obj.class_to_idx.items()}

    for element in dataset_obj:
        y_lbl = element[1]
        y_lbl = idx2class[y_lbl]
        count_dict[y_lbl] += 1

    return count_dict


# def get_class_distribution_loaders(dataloader_obj, dataset_obj):
#     count_dict = {k: 0 for k, v in dataset_obj.class_to_idx.items()}
#     idx2class = {v: k for k, v in dataset_obj.class_to_idx.items()}
#     for _, j in dataloader_obj:
#         y_idx = j.item()
#         y_lbl = idx2class[y_idx]
#         count_dict[str(y_lbl)] += 1
#
#     return count_dict


def get_class_distribution_loaders(dataloader_obj, dataset_obj,idx2class):
    count_dict = {k: 0 for k, v in dataset_obj.class_to_idx.items()}

    if dataloader_obj.batch_size == 1:
        for _, label_id in dataloader_obj:
            y_idx = label_id.item()
            y_lbl = idx2class[y_idx]
            count_dict[str(y_lbl)] += 1
    else:
        for _, label_id in dataloader_obj:
            for idx in label_id:
                y_idx = idx.item()
                y_lbl = idx2class[y_idx]
                count_dict[str(y_lbl)] += 1

    return count_dict


# print("Distribution of classes: \n", get_class_distribution(natural_img_dataset))


# def get_class_distribution(dataset_obj):
#     count_dict = {k: 0 for k, v in dataset_obj.class_to_idx.items()}
#
#     for _, label_id in dataset_obj:
#         label = idx2class[label_id]
#         count_dict[label] += 1
#     return count_dict


def plot_from_dict(dict_obj, plot_title, **kwargs):
    return sns.barplot(data=pd.DataFrame.from_dict([dict_obj]).melt(), x="variable", y="value", hue="variable",
                       **kwargs).set_title(plot_title)


def multi_acc(y_pred, y_test):
    # y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred, dim=1)
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc*100)

    return acc

#
# def multi_acc(y_pred, y_test):
#     # y_pred_softmax = torch.log_softmax(y_pred, dim=1)
#     _, y_pred_tags = torch.max(y_pred, dim=1)
#
#     correct_pred = (y_pred_tags == y_test).float()
#     acc = correct_pred.sum() / len(correct_pred)
#
#     acc = torch.round(acc) * 100
#
#     return acc
