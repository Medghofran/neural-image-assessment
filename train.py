import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import cuda 
import torch.autograd
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets
import torchvision.models as models

from torch.utils.data import DataLoader

from model import IMA, emd_loss
from dataset import AVADataset
from torch.utils.tensorboard import SummaryWriter, writer
import os

def main():
    # Check for GPU support 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()

    # Introduce data augmentation policies for training
    train_transforms = train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )]
    )

    # Introduce data augmentation policies for validation
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )]
    )

    # Create backbone 
    base_model = models.vgg16(pretrained=True)
    # Inatialize image assessment class with created backbone
    model = IMA(backbone = base_model)
    model = model.to(device)

    # Initialize the optimizer 
    conv_base_lr = 5e-3
    dense_lr = 5e-4
    optimizer = optim.SGD([
        {'params': model.features.parameters(), 'lr': conv_base_lr},
        {'params': model.classifier.parameters(), 'lr': dense_lr}],
        momentum=0.9
    )

    # Print number of learneable parameters
    param_num = 0
    for param in model.parameters():
        if param.requires_grad:
            param_num += param.numel()
    print('Trainable params: %.2f million' % (param_num / 1e6))

    # Setup train and validation datasets 
    img_path = './data/images'
    train_csv_file = './data/train_labels.csv'
    val_csv_file = './data/val_labels.csv' 
    trainset = trainset = AVADataset(csv_file=train_csv_file, root_dir=img_path, transform=train_transform)
    valset = AVADataset(csv_file=val_csv_file, root_dir=img_path, transform=val_transform)
    # Setup Torch data loaders 
    train_loader = DataLoader(trainset, batch_size=32,
        shuffle=True, num_workers=1
    )

    val_loader = DataLoader(valset, batch_size=32,
        shuffle=False, num_workers=1
    )

    # for early stopping
    count = 0
    init_val_loss = float('inf')
    train_losses = []
    val_losses = []

    epoch_count = 100
    for epoch in range(epoch_count):
        batch_losses = []
        for i, data in enumerate(train_loader):
            images = data['image'].to(device)
            labels = data['annotations'].to(device).float()
            outputs = model(images)
            outputs = outputs.view(-1, 10, 1)

            optimizer.zero_grad()

            loss = emd_loss(labels, outputs)
            batch_losses.append(loss.item())

            loss.backward()

            optimizer.step()

            print('Epoch: %d/%d | Step: %d/%d | Training EMD loss: %.4f' % (epoch + 1, epoch_count, i + 1, len(trainset) // 128 + 1, loss.data[0]))
            writer.add_scalar('batch train loss', loss.data[0], i + epoch * (len(trainset) // 128 + 1))

        avg_loss = sum(batch_losses) / (len(trainset) // 128 + 1)
        train_losses.append(avg_loss)
        print('Epoch %d mean training EMD loss: %.4f' % (epoch + 1, avg_loss))

        # exponetial learning rate decay
        decay = False
        lr_decay_rate = 0.95
        lr_decay_freq = 10
        if decay:
            if (epoch + 1) % 10 == 0:
                conv_base_lr = conv_base_lr * lr_decay_rate ** ((epoch + 1) / lr_decay_freq)
                dense_lr = dense_lr * lr_decay_rate ** ((epoch + 1) / lr_decay_freq)
                optimizer = optim.SGD([
                    {'params': model.features.parameters(), 'lr': conv_base_lr},
                    {'params': model.classifier.parameters(), 'lr': dense_lr}],
                    momentum=0.9
                )
        
        # do validation after each epoch
        batch_val_losses = []
        for data in val_loader:
            images = data['image'].to(device)
            labels = data['annotations'].to(device).float()
            with torch.no_grad():
                outputs = model(images)
            outputs = outputs.view(-1, 10, 1)
            val_loss = emd_loss(labels, outputs)
            batch_val_losses.append(val_loss.item())
        avg_val_loss = sum(batch_val_losses) / (len(valset) // 128 + 1)
        val_losses.append(avg_val_loss)
        print('Epoch %d completed. Mean EMD loss on val set: %.4f.' % (epoch + 1, avg_val_loss))
        writer.add_scalars('epoch losses', {'epoch train loss': avg_loss, 'epoch val loss': avg_val_loss}, epoch + 1)

        # Use early stopping to monitor training
        if avg_val_loss < init_val_loss:
            init_val_loss = avg_val_loss
            # save model weights if val loss decreases
            print('Saving model...')
            ckpt_path = './ckpts'
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)
            torch.save(model.state_dict(), os.path.join(ckpt_path, 'epoch-%d.pth' % (epoch + 1)))
            print('Done.\n')
            # reset count
            count = 0
        elif avg_val_loss >= init_val_loss:
            count += 1
            early_stopping_patience = 10
            if count == early_stopping_patience:
                print('Val EMD loss has not decreased in %d epochs. Training terminated.' % early_stopping_patience)
                break

        print('Training completed.')


if __name__ == "__main__":
    main()