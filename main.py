import argparse, os, sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision

from tensorboardX import SummaryWriter

# PATH = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, PATH + r'\utils')

from data_loader import load_dataloader
from models import load_model
from src.utils import *
from src.optimizer import make_optimizer
from src.scheduler import make_scheduler
from src.logger import get_logger
from train import train, valid

parser = argparse.ArgumentParser(description='Albu_MixMatch Pytorch')

parser.add_argument('--epochs', default=1024, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num-workers', default=0, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float)
parser.add_argument('--seed', default=2021, type=int)
parser.add_argument('--device', default='cuda')
parser.add_argument('--checkpoint-dir', type=str, default='checkpoint')
parser.add_argument('--checkpoint-name', type=str, default='')

parser.add_argument('--datasets', default='CIFAR10', choices=('CIFAR10', 'CIFAR100', 'SVHN', 'STL10'))
parser.add_argument('--num-classes', default=10, type=int)
parser.add_argument('--num-labels', default=400, type=int)
parser.add_argument('--evaluate', action='store_true', default=False)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--optimizer', default='ADAM', choices=('SGD', 'ADAM', 'ADAMW', 'RADAM', 'LOOKAHEAD'))
parser.add_argument('--decay-type', default='cosine_warmup', choices=('step', 'step_warmup', 'cosine_warmup'))

#Hyperparameters
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=75, type=float)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--K', default=2, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)

# logger = get_logger('RandAugment') ##used tqdm instead

def main():
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    l_train_loader, u_train_loader, val_loader, test_loader = load_dataloader(args)

    model = load_model(args)
    model = model.cuda()

    criterion = SemiLoss()
    eval_criterion = nn.CrossEntropyLoss().cuda()
    optimizer = make_optimizer(args, model)
    scheduler = make_scheduler(args, optimizer)

    if not os.path.isdir(args.checkpoint_dir):                                                           
        os.mkdir(args.checkpoint_dir)

    writer = SummaryWriter('result')
    best_loss = 10e10
    for epoch in range(args.epochs):
        trn_loss = train(args, model, l_train_loader, u_train_loader, criterion, optimizer, scheduler, epoch)
        vld_loss, vld_acc = valid(args, model, val_loader, eval_criterion, epoch)
        writer.add_scalar('losses/train_loss', trn_loss, epoch+1)
        writer.add_scalar('losses/valid_loss', vld_loss, epoch+1)
        writer.add_scalar('accs/valid_acc', vld_acc, epoch+1)

        if vld_loss < best_loss:
            save_checkpoint(model, args.checkpoint_dir, epoch+1, file_name='mixmatch.pt')
            best_loss = vld_loss

    
if __name__ == '__main__':
    main()