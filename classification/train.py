import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision as p
from my_dataset import MyDataset
from model import CNN
from loss import MyLoss
from alexnet import AlexNet
import torchvision
from resnet_cbam import resnet18
import argparse

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    # ============================ step 1/5 construct data loader =====================
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.8, 1.25)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.18,0.18,0.18], std=[0.24,0.24,0.24])

    ])

    valid_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.18,0.18,0.18], std=[0.24,0.24,0.24])
    ])

    train_data = MyDataset(data_dir=data_dir, train=True, transform=train_transform)
    valid_data = MyDataset(data_dir=data_dir, train=False, transform=valid_transform)

    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

    # ============================ step 2/5 define the model ============================
    if str(args.backbone) == "resnet18":
        net = torchvision.models.resnet18(pretrained=False)
    elif str(args.backbone) == "resnet18-pretrain":
        net = torchvision.models.resnet18(pretrained=True)
    elif str(args.backbone) == "alexnet":
        net = AlexNet(2)
    else:
        
        net = resnet18()
        model_path ='resnet18-5c106cde.pth' # 预训练参数的位置
        # 自己重写的网络
        model_dict = net.state_dict() # 网络层的参数
        # 需要加载的预训练参数
        print(torch.load(model_path).keys())
        pretrained_dict = torch.load(model_path)  # torch.load得到是字典，我们需要的是state_dict>下的参数
        pretrained_dict = {k.replace('module.', ''): v for k, v in
                               pretrained_dict.items()}  # 因为pretrained_dict得到module.conv1.weight，但是自己建的model无module，只是conv1.weight，所以改写下。

        # 删除pretrained_dict.items()中model所没有的东西
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # 只保留预练模型中，自己建的model有的参数
        model_dict.update(pretrained_dict)  # 将预训练的值，更新到自己模型的dict中
        net.load_state_dict(model_dict)  # model加载dict中的数据，更新网络的初始值
    
    net.fc = nn.Linear(net.fc.in_features,2,bias=False)
    net.to(device)

    # ============================ step 3/5 define the loss function ====================
    # criterion = MyLoss()
    #weight = torch.tensor([1.,2.,4.]).to(device)
    weight=None
    criterion = nn.CrossEntropyLoss(weight=weight)

    # ============================ step 4/5 define the optimizer ========================
    optimizer = optim.AdamW(net.parameters(), weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     T_max=MAX_EPOCH,
                                                     eta_min=0,
                                                     last_epoch=-1)

    # ============================ step 5/5 train the model =============================
    print('\nTraining start!\n')
    start = time.time()
    max_acc = 0.
    reached = 0  # which epoch reached the max accuracy
    loss_curve = []

    for epoch in range(1, MAX_EPOCH + 1):

        loss_mean = 0.

        net.train()
        for i, data in enumerate(train_loader):

            # forward
            inputs, labels = data['image'], torch.tensor(data['label'])
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)

            # backward
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()

            # update weights
            optimizer.step()

            # results
            _, predicted = torch.max(outputs.data, 1)

            # calculate the accuracy of this training iteration
            ### Begin your code ###
            train_acc = torch.sum(predicted == labels).item()/predicted.shape[0]

            ### End your code ###

            # print log
            loss_mean += loss.item()
            if (i+1) % log_interval == 0:
                loss_mean = loss_mean / log_interval
                loss_curve.append(loss_mean)
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH, i+1, len(train_loader), loss_mean, train_acc))
                loss_mean = 0.

        # validate the model
        if epoch % val_interval == 0:
            val_acc = 0
            loss_val = 0.
            net.eval()
            i = 0
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    inputs, labels = data['image'], torch.tensor(data['label'])
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    i += 1
                    loss_val += loss.item()

                # calculate the accuracy of the validation predictions
                ### Begin your code ###
                    val_acc += torch.sum(predicted == labels).item()/predicted.shape[0]
                ### End your code ###
                val_acc /= i
                if val_acc > max_acc:
                    max_acc = val_acc
                    reached = epoch
                    if args.save:
                        torch.save(net, os.path.join("model", "%s.pt" % str(args.backbone)))
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}\n".format(
                    epoch, MAX_EPOCH, j+1, len(valid_loader), loss_val, val_acc))

    print('\nTraining finish, the time consumption of {} epochs is {}s\n'.format(MAX_EPOCH, round(time.time() - start)))
    print('The max validation accuracy is: {:.2%}, reached at epoch {}.\n'.format(max_acc, reached))
    print(loss_curve)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="COVID19 Training Pipline")
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet18-pretrain', 'resnet18-cbam','alexnet'],
                        help='backbone name (default: resnet18)')
    parser.add_argument('--save', type=bool, default=False,
                        help="choose to save model or not.")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("\nRunning on:", device)

    if device == 'cuda':
        device_name = torch.cuda.get_device_name()
        print("The device name is:", device_name)
        cap = torch.cuda.get_device_capability(device=None)
        print("The capability of this device is:", cap, '\n')
    
    # hyper-parameters
    seed = 1
    MAX_EPOCH = 10
    BATCH_SIZE = 60
    LR = 0.001
    weight_decay = 1e-4
    log_interval = 2
    val_interval = 1
    data_dir = "Dataset_BUSI"
    
    set_seed(seed)
    print('random seed:', seed)
    main()









