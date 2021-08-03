import time
import torch
import torch.nn.functional as F

from tqdm import tqdm
from src.utils import AverageMeter, interleave, sharpen, mixup, accuracy

def train(args, model, l_train_loader, u_train_loader, criterion, optimizer, scheduler, epoch):
    epoch_start = time.time()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    l_train_loader_iter  = iter(l_train_loader)
    u_train_loader_iter = iter(u_train_loader)
    n_iters = 1024
    tq = tqdm(range(n_iters), total = n_iters, leave=True)
    model.train()
    for step in tq:
        try:
            imgs, targs = l_train_loader_iter.next()
        except: #labeliter reinitialize
            l_train_loader_iter  = iter(l_train_loader)
            imgs, targs = l_train_loader_iter.next()
        try:
            uimgs, _ = u_train_loader_iter.next()
        except: #unlabeliter reinitialize
            u_train_loader_iter = iter(u_train_loader)
            uimgs, _ = u_train_loader_iter.next()
        imgs, targs = imgs.to(args.device), targs.type(torch.LongTensor).to(args.device)
        targs = F.one_hot(targs, args.num_classes)
        uimgs = [uimg.type(torch.FloatTensor).to(args.device) for uimg in uimgs]

        with torch.no_grad():
            outputs = [model(uimg) for uimg in uimgs]
            p_model = [torch.softmax(output, dim=1) for output in outputs]
            q_bar = torch.sum(torch.stack(p_model), dim=0) / args.K
            utargs = sharpen(q_bar, args.T)
            utargs.detach_()

        all_imgs = torch.cat([imgs]+uimgs, dim=0)
        all_targs = torch.cat([targs]+[utargs]*args.K, dim=0)
        
        mixup_imgs, mixup_targs = mixup(args, all_imgs, all_targs)
        logits = [model(mixup_img) for mixup_img in mixup_imgs]
        logits = interleave(logits, args.batch_size)

        assert len(logits)==3 and logits[0].shape == (args.batch_size, args.num_classes)

        Lx, Lu, lam_u = criterion(args, logits, all_targs, epoch)
        loss = Lx + lam_u * Lu
        losses.update(loss)
        losses_x.update(Lx)
        losses_u.update(Lu)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0:
            tq.set_description('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                                            epoch+1, args.epochs, step+1, n_iters, losses.avg))
    scheduler.step()
    tq.write(f'----------------Epoch {epoch+1} train finished------------------')
    tq.write('Epoch [{}/{}], Time elapsed: {:.4f}s, Loss: {:.4f}, Loss_x: {:.4f}, Loss_u: {:.4f}, lr: {:.4f}'.format(
        epoch+1, args.epochs, time.time()-epoch_start, losses.avg, losses_x.avg, losses_u.avg, scheduler.get_lr()[0]))
    return losses.avg

def valid(args, model, val_loader, criterion, epoch):
    epoch_start = time.time()
    losses = AverageMeter()
    accs = AverageMeter()
    model.eval()
    # tq = tqdm(val_loader, total=len(val_loader), leave=False)
    tq = tqdm(enumerate(val_loader), total=len(val_loader), leave=False)
    with torch.no_grad():
        for step, (images, targets) in tq:
            images, targets = images.type(torch.FloatTensor).to(args.device), targets.type(torch.LongTensor).to(args.device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            acc = accuracy(outputs, targets)[0]
            accs.update(acc)
            losses.update(loss)
    tq.write(f'----------------Epoch {epoch+1} validation finished------------------')
    tq.write('Epoch [{}/{}], Time elapsed: {:.4f}s, Loss: {:.4f}, Accuracy: {:.2f}%'.format(
        epoch+1, args.epochs, time.time()-epoch_start, losses.avg, accs.avg))
    return losses.avg, accs.avg