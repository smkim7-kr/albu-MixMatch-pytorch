import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random, os

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class AverageMeter(object):
    '''
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, largest=True, sorted=True)  
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class SmoothCrossEntropyLoss(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        if len(target.size()) == 1:
            target = torch.nn.functional.one_hot(target, num_classes=input.size(-1))
            target = target.float().cuda()
        if self.label_smoothing > 0.0:
            s_by_c = self.label_smoothing / len(input[0])
            smooth = torch.zeros_like(target)
            smooth = smooth + s_by_c
            target = target * (1. - s_by_c) + smooth

        return torch.nn.cross_entropy(input, target)

class SemiLoss(object):
    def __call__(self, args, logits, targets, epoch):
        logits_x, logits_u = logits[0], torch.cat(logits[1:], dim=0)        
        targets_x, targets_u = targets[:args.batch_size], targets[args.batch_size:]
        probs_u = torch.softmax(logits_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, args.lambda_u * float(np.clip(epoch / args.epochs, 0.0, 1.0))

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

def mixup(args, all_imgs, all_targets):
    beta = np.random.beta(args.alpha, args.alpha)
    beta = max(beta, 1 - beta)
    idx = torch.randperm(all_imgs.size(0))

    img_a, img_b = all_imgs, all_imgs[idx]
    target_a, target_b = all_targets, all_targets[idx]
    img_mixup = beta * img_a + (1 - beta) * img_b
    target_mixup = beta * target_a + (1 - beta) * target_b

    img_mixup = list(torch.split(img_mixup, args.batch_size))
    img_mixup = interleave(img_mixup, args.batch_size)
    return img_mixup, target_mixup

def sharpen(x, T):
    sharp = x ** (1 / T)
    return sharp / sharp.sum(dim=1, keepdim=True)

def save_checkpoint(model, saved_dir, epoch, file_name):
    file_name = "Epoch" + str(epoch) + "_" + file_name
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model, output_path)