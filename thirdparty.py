# For data preprocess
import random
import numpy as np
import pandas as pd
import csv
import os
import glob
import json
import math
import cv2
# For feature selection
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
# PyTorch
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,ConcatDataset, Subset
from torchvision.datasets import DatasetFolder,ImageFolder
from torchvision import transforms,datasets
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
# For plotting and image
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from PIL import Image
from pathlib import Path
# # tqdm bar
from tqdm.auto import tqdm


#设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

#获取设备名称
def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'

#画曲线图
def plot_curve(x,y,label=None,title='',xlabel='',ylabel='',xlim=[0,0],ylim=[0,0]):
    figure(figsize=(6, 4))
    color=['tab:red','tab:green','tab:blue','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    for i in range(len(x)):
        plt.plot(x[i], y[i], c=color[i], label=label[i])#画出每次训练损失函数的大小
    #画图函数的颜色 tab:blue,tab:orange,tab:green,tab:red,tab:purple,tab:brown,tab:pink,tab:gray,tab:olive,tab:cyan
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('{}'.format(title))
    plt.legend()
    plt.show()

#画训练曲线图
def plot_train_curve(loss_record, title='',ylim=[0,1]):
    total_steps = len(loss_record['train'])
    x_1 = np.arange(total_steps)
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')#画出每次训练损失函数的大小
    if len(loss_record['val'])!=0:#每一个epoch过后会看有没有验证集上的损失函数
        x_2 = x_1[::len(loss_record['train']) // len(loss_record['val'])]
        plt.plot(x_2, loss_record['val'], c='tab:cyan', label='val')
        #画图函数的颜色 tab:blue,tab:orange,tab:green,tab:red,tab:purple,tab:brown,tab:pink,tab:gray,tab:olive,tab:cyan
    plt.ylim(ylim[0],ylim[1])
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()

#画散点图并画对角线
def plot_scatter(x,y,label=None,title='',xlabel='',ylabel='',xlim=[0,0],ylim=[0,0]):
    ''' Plot scatter'''
    figure(figsize=(6, 4))
    color = ['tab:red', 'tab:green', 'tab:blue', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive',
             'tab:cyan']
    for i in range(len(x)):
        plt.scatter(x[i], y[i], c=color[i], label=label[i])  # 画出每次训练损失函数的大小
    # 画图函数的颜色 tab:blue,tab:orange,tab:green,tab:red,tab:purple,tab:brown,tab:pink,tab:gray,tab:olive,tab:cyan
    plt.plot([xlim[0], xlim[1]], [ylim[0], ylim[1]])
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('{}'.format(title))
    plt.legend()
    plt.show()

#图片显示
def image_show(image,color_type="rgb",title="Example Image"):
    shape_info=image.shape
    if(image.dtype!=np.uint8):
        image = (image * 255).astype(np.uint8)
    if(shape_info[0] == 1 or shape_info[0] == 3):
        image = np.transpose(image, (1, 2, 0))
    if (color_type=="bgr"):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if (color_type=="gray"):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')  # 关闭坐标轴
    plt.show()
#变换成torch数据
def torch_data(data,device="cpu"):
    if isinstance(data, torch.Tensor): return data
    else:
        data = torch.tensor(data,device=device)
    #if(data.dim()==1): data=data.unsqueeze(dim=1)
    return data

def torch_label(label,device="cpu",mode="regression"):
    if isinstance(label, torch.Tensor):
        if (mode == "regression"):
            label = label.detach().to(dtype=torch.float32, device=device)
        elif (mode == "classification"):
            label = label.detach().to(dtype=torch.int64, device=device)
    else:
        if (mode == "regression"):
            label = torch.tensor(label, dtype=torch.float32, device=device)
        elif (mode == "classification"):
            label = torch.tensor(label, dtype=torch.int64, device=device)
    if (label.dim() == 1 and mode=="regression"):
        label = label.unsqueeze(dim=1)
    if (label.dim() == 2 and mode == "classification"):
        label = label.squeeze(dim=1)
    return label

def get_pseudo_labels(unlabel_set, model, config, device, mode,threshold=0.85,to_torch=True):
    total_unlabel = len(unlabel_set.dataset)
    # 模型是在eval模式下
    model.eval()
    softmax = nn.Softmax(dim=-1)
    masks = []
    pseudo_labels = []
    print("getting pseudo_labels...")
    for data in tqdm(unlabel_set):
        data = torch_data(data, device)
        with torch.no_grad():
            logits = model(data)
        probs = softmax(logits).cpu()
        preds = torch.max(probs, 1)[1]
        mask = torch.max(probs, 1)[0] > threshold
        masks.append(mask)
        pseudo_labels.append(preds)
    mask = torch.cat(masks, dim=0).cpu().numpy()
    pseudo_label = torch.cat(pseudo_labels, dim=0).cpu().numpy()
    indices = torch.arange(0, total_unlabel)[mask]
    unlabel_dataset = unlabel_set.dataset
    pseudo_dataset = PseudoDataset(unlabel_dataset,indices,pseudo_label,to_torch)
    print("using {0:.2f}% unlabeld data,number is {1:7d}".format(100 * len(pseudo_dataset) / total_unlabel,len(indices)))
    return pseudo_dataset
#
def get_cosine_schedule_with_warmup(optimizer: Optimizer,warmup_steps: int,total_steps: int,num_cycles: float = 0.5,last_epoch: int = -1,):
  def lr_lambda(current_step):
    # Warmup
    if current_step < warmup_steps:
      return float(current_step) / float(max(1, warmup_steps))
    # decadence
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
  return LambdaLR(optimizer, lr_lambda, last_epoch)
#训练函数，分为回归和分类
def train(train_set, val_set, model, config, device,mode="regression",do_semi=False,unlabel_set=None,
          semi_threhold = 0, param_group=True,param_learn_ratio=0.01):
    ''' Model training '''
    n_epochs = config['n_epochs']  # Maximum number of epochs
    batch_show_step,epoch_show_step=config["batch_show_step"],config["epoch_show_step"]
    if param_group:
        if(param_learn_ratio<1e-6):
            for name, param in model.module.net.named_parameters():
                if name not in ["fc.weight","fc.bias"]:
                    param.requires_grad=False
            optimizer = getattr(torch.optim, config['optimizer'])(model.module.parameters(), **config['optim_hparas'])
        else:
            params_1x = [param for name, param in model.module.net.named_parameters() if name not in ["fc.weight","fc.bias"]]
            optimizer = getattr(torch.optim, config['optimizer'])([{
            'params': params_1x,'lr': config['optim_hparas']['lr'] * param_learn_ratio}, {'params': model.module.net.fc.parameters(),'lr': config['optim_hparas']['lr']}]
            , **config['optim_hparas'])
    else:
        optimizer = getattr(torch.optim, config['optimizer'])(model.module.parameters(), **config['optim_hparas'])
    # Setup scheduler
    sche_flag = False
    if('scheduler' in config):
        scheduler = get_cosine_schedule_with_warmup(optimizer, **config['sche_hparas'])
        sche_flag = True
    min_loss = 1000.
    loss_record = {'train': [], 'val': []}
    acc_record = {'train': [], 'val': []}
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        if do_semi:
            train_dataset = train_set.dataset
            _ , label_example = train_dataset[0]
            to_torch = True if isinstance(label_example, torch.Tensor) else False
            pseudo_dataset = get_pseudo_labels(unlabel_set, model, config,device,mode,threshold=semi_threhold,to_torch=to_torch)
            concat_dataset = ConcatDataset([train_dataset, pseudo_dataset])
            train_set = DataLoader(concat_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=True, drop_last=False)
        model.train()
        step = 0
        show_flag=False
        if((epoch + 1) % epoch_show_step == 1 or epoch_show_step == 1): show_flag=True
        for data, label in train_set:  # iterate through the dataloader
            data = torch_data(data,device)
            label = torch_label(label,device,mode)
            pred = model(data)  # forward pass (compute output)
            loss = model.module.cal_loss(pred, label, config['l_lambda'], config['regular_type'])  # compute loss
            loss.backward()  # compute gradient (backpropagation)
            optimizer.step()  # update model with optimizer
            if(sche_flag): scheduler.step()
            optimizer.zero_grad()  # set gradient to zero
            train_acc=0
            if(mode == "classification"):
                _, pred = torch.max(pred, 1)  # get the index of the class with the highest probability
                train_acc = ((pred.cpu() == label.cpu()).sum().item()) * 1.0 / len(data)
                acc_record['train'].append(train_acc)
            train_loss = loss.detach().cpu().item()
            loss_record['train'].append(train_loss)
            step += 1
            if(show_flag):
                if (step % batch_show_step == 1 or batch_show_step==1):
                    if(mode=="classification"):
                        print('epoch{:4d}|step = {:4d}, train_loss = {:.4f}, train_accuracy = {:.2%}'.format(epoch + 1, step,train_loss,train_acc))
                    if (mode == "regression"):
                        print('epoch{:4d}|step = {:4d}, train_loss = {:.4f}'.format(epoch + 1, step,train_loss))
        val_loss, val_acc = val(val_set, model, config, device,mode)
        if (show_flag):  # val_cross < min_cross
            # Save model if your model improved
            if(mode == "classification"):
                print("epoch = {:4d}, val_loss = {:.4f}, val_accuracy = {:.2%}".format(epoch + 1, val_loss,val_acc))
            else:
                print("epoch = {:4d}, val_loss = {:.4f}".format(epoch + 1, val_loss))
        if(val_loss < min_loss):
            min_loss = val_loss
            torch.save(model.state_dict(), config['save_path'])
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
        epoch += 1
        loss_record['val'].append(val_loss)
        if (mode == "classification"):
            acc_record['val'].append(val_acc)
        if early_stop_cnt > config['early_stop']:
            break
    print('Finished training after {} epochs'.format(epoch))
    return min_loss, loss_record, acc_record

#验证函数，分为回归和分类
def val(val_set, model, config, device, mode="regression"):
    model.eval()  # set model to evalutation mode
    total_loss = 0
    total_acc = 0
    for data, label in tqdm(val_set):  # iterate through the dataloader
        data = torch_data(data, device)
        label = torch_label(label, device, mode)
        with torch.no_grad():  # disable gradient calculation
            pred = model(data)  # forward pass (compute output)
            loss = model.module.cal_loss(pred, label, config['l_lambda'], config['regular_type'])  # compute loss
            if (mode == "classification"):
                _, pred = torch.max(pred, 1)  # get the index of the class with the highest probability
                total_acc += (pred.cpu() == label.cpu()).sum().item()
        total_loss += loss.detach().cpu().item() * len(data)  # accumulate loss
    total_loss = total_loss / len(val_set.dataset)  # compute averaged loss
    total_acc = total_acc / len(val_set.dataset)
    return total_loss, total_acc

#验证函数，分为回归和分类
def test(test_set, model, device, mode="regression"):
    model.eval()  # set model to evalutation mode
    preds = []
    for data in tqdm(test_set):  # iterate through the dataloader
        data = torch_data(data, device)
        with torch.no_grad():  # disable gradient calculation
            pred = model(data)  # forward pass (compute output)
            if (mode == "classification"):
                _, pred = torch.max(pred, 1)  # get the index of the class with the highest probability
            preds.append(pred.detach().cpu())  # collect prediction
    preds = torch.cat(preds, dim=0)  # concatenate all predictions and convert to a numpy array
    return preds

#标准化和归一化
def self_normalize(tensor, mean=None, std=None):
    if ( mean == None):
        mean = torch.mean(tensor,dim=0)
    if ( std == None):
        std = torch.std(tensor, dim=0)
    return (tensor - mean)*1.0 / std,mean,std
def self_reverse_normalize(tensor, mean, std):
    return tensor*std*1.0+mean
def self_minmax(tensor, maxval=None, minval=None):
    if ( maxval == None):
        maxval,maxindex = torch.max(tensor,dim=0)
    if ( minval == None):
        minval,minindex = torch.min(tensor, dim=0)
    return (tensor - minval)*1.0 / (maxval - minval),maxval,minval
def self_reverse_minmax(tensor, maxval, minval):
    return tensor*(maxval - minval)*1.0 + minval

#图像变换
train_tfm_rgb = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    # You may add some transforms here.
    transforms.RandomRotation(40),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.2),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((224, 224)),
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

show_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class SelfDataset(Dataset):
    ''' Dataset for loading and preprocessing the COVID19 dataset '''
    def __init__(self,data,label=None,mode='train',device="cpu"):
        self.mode = mode
        self.data = data
        self.label = label
        self.dim = self.data.shape[1]
        print('Finished reading the {} set of Dataset ({} samples found, each dim = {})'.format(mode, len(self.data), self.dim))
    def __getitem__(self, index):
        if index < 0:  # Handle negative indices
            index += len(self)
        if index >= len(self):
            raise IndexError("index %d is out of bounds for axis 0 with size %d" % (index, len(self)))
        # Returns one sample at a time
        if(self.mode =="train" or self.mode =="val"):
            return self.data[index],self.label[index]
        else:
            return self.data[index]
    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)

class SelfImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_names = []
        # 遍历所有子文件夹，收集图片文件
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # 根据需要添加支持的图片格式
                    self.img_names.append(os.path.join(subdir, file))
    def __len__(self):
        """返回数据集中图片的总数"""
        return len(self.img_names)
    def __getitem__(self, idx):
        """根据索引读取图片"""
        img_path = self.img_names[idx]
        image = Image.open(img_path).convert('RGB')  # 读取图片并转换为RGB模式
        if self.transform:
            image = self.transform(image)
        return image

class PseudoDataset(Dataset):
    def __init__(self, unlabeled_set, indices, pseudo_labels,to_torch):
        self.data = Subset(unlabeled_set, indices)
        if(to_torch):
            self.label = torch.tensor(pseudo_labels,dtype=torch.int64)[indices]
        else:
            self.label = pseudo_labels[indices]
    def __getitem__(self, index):
        if index < 0 : #Handle negative indices
            index += len(self)
        if index >= len(self):
            raise IndexError("index %d is out of bounds for axis 0 with size %d"%(index, len(self)))
        x = self.data[index]
        y = self.label[index]
        return x, y
    def __len__(self):
        return len(self.data)
