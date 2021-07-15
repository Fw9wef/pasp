import os

import numpy as np
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn

from model import MRZdetector
from eval import evaluate

from utils.logger import Logger
from utils.datasets import ListDataset
from utils.utils import adjust_learning_rate, clip_gradient, rev_label_map, device, show_metrics

from tqdm import tqdm

# Data parameters
save_folder = "./checkpoints_adam"
data_folder = "./annotation"  # folder with data files
log_folder = "logs"
os.makedirs(save_folder, exist_ok=True)

# Model parameters
# Learning parameters
checkpoint = None  # path to model checkpoint("./best/52.pth.tar" ), None if none
batch_size = 32  # batch size
start_epoch = 0  # start at this epoch
epochs = 200  # number of epochs to run without early-stopping
epochs_since_improvement = 0  # number of epochs since there was an improvement in the validation metric
min_f1 = 0.6  # min avg value f1 on test sample for saving model
workers = 16  # number of workers for loading data in the DataLoader
# Show result
print_ascii_table = True
adjust_lr_flag = True
# clip if gradients are exploding, which may happen at larger
# batch sizes (sometimes at 32) - you will recognize it by
# a sorting error in the MuliBox loss calculation
grad_clip = None
lr = 0.5e-3  # learning rate
momentum = 0.9  # momentum
weight_decay = 5e-5  # weight decay
opt = 'adam'  # optimizer

# Initiate logger
logger = Logger(log_folder)

cudnn.benchmark = True


def main():
    """
    Training and validation.
    """
    global checkpoint, min_f1
    min_loss = np.inf

    model = MRZdetector()

    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint)

    # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
    biases = list()
    not_biases = list()
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
            else:
                not_biases.append(param)

    if opt == "sgd":
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                     lr=lr,
                                     weight_decay=weight_decay)

    # Move to default device
    model = model.to(device)
    criterion = torch.nn.MSELoss().to(device)

    # Custom DataLoaders
    train_dataset = ListDataset(data_folder, split='train')
    test_dataset = ListDataset(data_folder, split='test')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2 * batch_size, shuffle=False,
                                              num_workers=workers, pin_memory=True)

    num_batches = len(train_loader)

    best_f1 = min_f1  # assume a threshold f1 at first

    # Epochs
    for epoch in range(start_epoch, epochs):

        if adjust_lr_flag:
            adjust_learning_rate(optimizer, epoch, epochs, lr)

        batches_done = num_batches * epoch

        train_dataset.select_train_data()

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              num_iter=batches_done)

        torch.cuda.empty_cache()

        # One epoch's validation
        loss = evaluate(model=model, test_loader=test_loader)
        e_loss = loss.mean()

        # Did validation f1 improve?
        if e_loss < min_loss:
            print("Saving model ...")
            min_loss = e_loss
            torch.save(model.state_dict(), os.path.join(save_folder, f"BEST.pth"))
        torch.save(model.state_dict(), os.path.join(save_folder, f"{epoch}.pth"))
        torch.cuda.empty_cache()


def train(train_loader, model, criterion, optimizer, num_iter):
    model.train()  # training mode enables dropout

    # Batches
    for i, (images, _, target) in enumerate(tqdm(train_loader, desc='Training')):

        num_iter += 1
        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)

        gt_xywh = target[:, :4]
        gt_sc = target[:, -3:-1]
        target = torch.cat([gt_xywh, gt_sc], dim=-1)
        target = target.to(device)

        # Forward prop.
        pred = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(pred, target)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()


if __name__ == '__main__':
    main()
