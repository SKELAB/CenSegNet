import os
import numpy as np
from PIL import Image
import torch
import time
import torchvision
from torch.utils.data import Dataset, DataLoader
import cv2
import random
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from glob import glob
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as ff
import torchvision.transforms as transforms
from torch import nn
from torch import optim
from torchvision import models
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from deepLabv3 import DeepLabv3_plus
# from metrics_file import eval_semantic_segmentation
import nibabel as nib
from sklearn.metrics import confusion_matrix
# from segnet import SegNet
# from metrics_file import eval_semantic_segmentation
import matplotlib.pyplot as plt

class SegmentationDataset(Dataset):
    def __init__(self, data, transform=None, grayscale=False):

        self.data = data
        self.transform = transform
        self.grayscale = grayscale

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx][0]
        mask_path = self.data[idx][1]

        image = cv2.imread(image_path)
        label = cv2.imread(mask_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        if label.max() > 1:
            label = label / 255
        label = label.reshape(1, label.shape[0], label.shape[1])
        if self.grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.transform:
            image = self.transform(image)
        return image, label

save_dir= '../Pre_process'
# final_iridis_total_patches_train
with open(os.path.join(save_dir,'final_iridis_total_patches_train.pkl'), 'rb') as f:
    train_list = pickle.load(f)

test_size = 0.2
# Split the data into training and tesbt sets
train_data, test_data = train_test_split(train_list, test_size=test_size, random_state=42)


transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
rgb=True
if rgb:
    train_dataset = SegmentationDataset(train_data[0:-1], transform=transform, grayscale=False)
    valid_dataset = SegmentationDataset(test_data, transform=transform, grayscale=False)
else:
    train_dataset = SegmentationDataset(train_data, transform=transform, grayscale=True)
    valid_dataset = SegmentationDataset(test_data, transform=transform, grayscale=True)


batch_size = 16
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader= torch.utils.data.DataLoader(valid_dataset, batch_size=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepLabv3_plus(nInputChannels=3, n_classes=1, os=16, _print=False).to(device)
# criterion = nn.NLLLoss().to(device)
criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(segnet.parameters(), lr=1e-5)
lr = 1e-4
optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
best_loss = float('inf')

num_class = 2
train_losses = []
valid_losses = []

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    start_epoch_time = time.time()
    running_train_loss = 0
    # train_acc = 0
    # train_miou = 0
    # train_dice = 0
    # train_class_acc = 0
    # error = 0
    for i, (image, label) in enumerate(train_dataloader):
        start_time = time.time()
        image = image.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)
        pred = model(image)
        loss = criterion(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

        # pre_label = pred.max(dim=1)[1].data.cpu().numpy()
        # true_label = label.data.cpu().numpy()
        # eval_metrix = eval_semantic_segmentation(pre_label, true_label)
        # train_acc += eval_metrix['pixel_accuracy']
        # train_miou += eval_metrix['miou']
        # if len(eval_metrix['class_accuracy']) < num_class:
        #     eval_metrix['class_accuracy'] = 0
        #     train_class_acc = train_class_acc + eval_metrix['class_accuracy']
        #     error += 1
        # else:
        #     train_class_acc = train_class_acc + eval_metrix['class_accuracy']
        # train_dice += eval_metrix['dice']

        end_time = time.time()  # End time for the batch
        batch_time = end_time - start_time
        if i % 20 == 0:
            print('|Epoch [{}/{}]|batch[{}/{}]|batch_loss:{:.9f}||batch_time:{:.4f}s'
                  .format(epoch,
                          num_epochs,
                          i + 1,
                          len(train_dataloader), loss.item(), batch_time))

        epoch_train_loss = running_train_loss / len(train_dataloader)
        train_losses.append(epoch_train_loss)
    end_epoch_time = time.time()  # End time for the epoch
    epoch_time = end_epoch_time - start_epoch_time  # Time taken for the whole epoch
    print(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.4f}s")

    # metric_description = '|Train Acc|: {:.5f}\n|Train dice|: {:.5f}\n|Train Mean IoU|: {:.5f}\n|Train_class_acc|: {:}'.format(
    #     train_acc / len(train_dataloader),
    #     train_dice / len(train_dataloader),
    #     train_miou / len(train_dataloader),
    #     train_class_acc / (len(train_dataloader)-error))

    model.eval()
    running_valid_loss = 0.0
    eval_loss = 0
    # eval_acc = 0
    # eval_miou = 0
    # eval_class_acc = 0
    # eval_dice = 0
    # # error = 0
    with torch.no_grad():
        for image, label in test_dataloader:
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            pred = model(image)
            loss = criterion(pred, label)
            running_valid_loss += loss.item()

            # pre_label = pred.max(dim=1)[1].data.cpu().numpy()
            # true_label = label.data.cpu().numpy()
            # eval_metrics = eval_semantic_segmentation(pre_label, true_label)
            # eval_acc = eval_metrics['pixel_accuracy'] + eval_acc
            # eval_miou = eval_metrics['miou'] + eval_miou

            # eval_class_acc =  eval_metrics['class_accuracy'] + eval_class_acc
            # eval_dice = eval_metrics['dice'] + eval_dice

        epoch_valid_loss = running_valid_loss / len(test_dataloader)
        valid_losses.append(epoch_valid_loss)
        # val_str = '|val Acc|: {:.5f}\n|val dice|: {:.5f}\n|val Mean IoU|: {:.5f}\n|val_class_acc|: {:}'.format(
        # eval_acc / len(test_dataloader),
        # eval_dice / len(test_dataloader),
        # eval_miou / len(test_dataloader),
        # eval_class_acc / (len(test_dataloader)-error))

    print("*" * 20)
    print(
        f"Epoch [{epoch + 1}] Train Loss: {epoch_train_loss:.4f} | Validation Loss: {epoch_valid_loss:.4f}")
    # print(val_str)
    print("*" * 20)

    if epoch_valid_loss < best_loss:
        best_loss = epoch_valid_loss
        torch.save(model.state_dict(), './records/best_deeplabv3_{}.pth'.format(epoch))

np.save('./losses/train_loss_deeplabv3.npy', train_losses)
np.save('./losses/valid_loss_deeplabv3.npy', valid_losses)

train_loss = np.load('./losses/train_loss_deeplabv3.npy', allow_pickle=True)
train_loss = train_loss.tolist()
valid_loss = np.load('./losses/valid_loss_deeplabv3.npy', allow_pickle=True)
valid_loss = valid_loss.tolist()
# Create a plot with labels and a legend
plt.plot(train_loss, label='train_loss', linestyle='-', marker='o', markersize=4)
plt.plot(valid_loss, label='valid_loss', linestyle='--', marker='s', markersize=4)

# Add titles and labels
plt.title('Loss')
plt.xlabel('Epoches')
plt.ylabel('Loss')

# Add a legend
plt.legend()

# Customize the gridlines
plt.grid(True, linestyle='--', alpha=0.6)

# Customize the tick marks and font size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
plt.savefig("./losses/deeplabv3_result.png")
