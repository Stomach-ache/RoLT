from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import sys
import argparse
import numpy as np
from InceptionResNetV2 import *
from KNNClassifier import KNNClassifier
from sklearn.mixture import GaussianMixture
import dataloader_webvision as dataloader
import torchnet
import torch.multiprocessing as mp
import time

parser = argparse.ArgumentParser(description='PyTorch WebVision Parallel Training')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--id', default='',type=str)
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid1', default=0, type=int)
parser.add_argument('--gpuid2', default=1, type=int)
parser.add_argument('--num_class', default=50, type=int)
parser.add_argument('--data_path', default='./webvision/', type=str, help='path to dataset')
parser.add_argument('--feat_size', default=1536, type=int)
parser.add_argument('--resume', default=-1, type=int)
parser.add_argument('--resume_id', default='', type=str)
parser.add_argument('--baseline', action='store_true')
parser.add_argument('--cls_ind_dist', action='store_true')
parser.add_argument('--cls_ind_loss', action='store_true')
parser.add_argument('--imb_ratio', default=1, type=float)
parser.add_argument('--warm_up', default=1, type=int)
parser.add_argument('--drw', action='store_true')
parser.add_argument('--drw_epoch', default=81, type=int)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '%s,%s'%(args.gpuid1,args.gpuid2)
random.seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
cuda1 = torch.device('cuda:0')
cuda2 = torch.device('cuda:1')


logger_name = ''

# Training
def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader,device,whichnet):
    criterion = SemiLoss()

    net.train()
    net2.eval() #fix one network and train the other


    cls_num_list = labeled_trainloader.dataset.img_num_list

    idx = epoch // args.drw_epoch
    betas = [0, 0.9999]
    effective_num = 1.0 - np.power(betas[idx], cls_num_list)
    per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)


    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        batch_size = inputs_x.size(0)

        single_labels_x = labels_x.clone()

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)
        w_x = w_x.view(-1,1).type(torch.FloatTensor)

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.to(device,non_blocking=True), inputs_x2.to(device,non_blocking=True), labels_x.to(device,non_blocking=True), w_x.to(device,non_blocking=True)
        inputs_u, inputs_u2 = inputs_u.to(device), inputs_u2.to(device)


        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)

            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4
            ptu = pu**(1/args.T) # temparature sharpening

            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()

            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)

            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px
            ptx = px**(1/args.T) # temparature sharpening

            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize
            targets_x = targets_x.detach()


        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1-l)

        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a[:batch_size*2] + (1 - l) * input_b[:batch_size*2]
        mixed_target = l * target_a[:batch_size*2] + (1 - l) * target_b[:batch_size*2]


        all_labels_x = single_labels_x.repeat(2)
        weights_x = per_cls_weights[all_labels_x]
        #print (weights_x.shape, idx.shape)
        mixed_weights = weights_x
        #mixed_weights = l * weights_x + (1 - l) * weights_x[idx[:batch_size*2]]
        mixed_weights = mixed_weights.to(device)

        logits = net(mixed_input)


        if args.drw:
            Lx = -torch.sum(mixed_weights * torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1)) / torch.sum(mixed_weights)
        else:
            Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))

        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.to(device)
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write('\n')
        sys.stdout.write('%s |%s Epoch [%3d/%3d] Iter[%4d/%4d]\t Labeled loss: %.2f'
                %(args.id, whichnet, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item()))
        sys.stdout.flush()

def warmup(epoch,net,optimizer,dataloader,device,whichnet):
    CEloss = nn.CrossEntropyLoss()
    acc_meter = torchnet.meter.ClassErrorMeter(topk=[1,5], accuracy=True)

    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):

        inputs, labels = inputs.to(device), labels.to(device,non_blocking=True)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = CEloss(outputs, labels)

        #penalty = conf_penalty(outputs)
        L = loss #+ penalty

        L.backward()
        optimizer.step()

        sys.stdout.write('\n')
        sys.stdout.write('%s |%s  Epoch [%3d/%3d] Iter[%4d/%4d]\t CE-loss: %.4f'
                %(args.id, whichnet, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()


def test(epoch,net1,net2,test_loader,device,queue):
    acc_meter = torchnet.meter.ClassErrorMeter(topk=[1,5], accuracy=True)
    acc_meter.reset()
    net1.eval()
    net2.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device,non_blocking=True)
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)
            acc_meter.add(outputs,targets)
    accs = acc_meter.value()
    queue.put(accs)


def eval_train_baseline(eval_loader,model,device,whichnet,queue):
    CE = nn.CrossEntropyLoss(reduction='none')
    model.eval()
    num_iter = (len(eval_loader.dataset)//eval_loader.batch_size)+1
    losses = torch.zeros(len(eval_loader.dataset))
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.to(device), targets.to(device,non_blocking=True)
            outputs = model(inputs)
            loss = CE(outputs, targets)
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]
            #sys.stdout.write('\n')
            #sys.stdout.write('|%s Evaluating loss Iter[%3d/%3d]\t' %(whichnet,batch_idx,num_iter))
            #sys.stdout.flush()

    losses = (losses-losses.min())/(losses.max()-losses.min())

    start_time = time.time()

    # fit a two-component GMM to the loss
    input_loss = losses.reshape(-1,1)
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=1e-3)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:,gmm.means_.argmin()]

    time_elapsed = time.time() - start_time
    log_log=open('./checkpoint/%s'%(args.id)+'_log.txt','a')
    log_log.write('GMM time elapsed (seconds): %.2f\n'%(time_elapsed))
    log_log.close()

    queue.put(prob)

def eval_train_rolt(eval_loader,model,device,whichnet,queue):
    model.eval()

    total_features = torch.empty((0, args.feat_size)).to(device)
    total_labels = torch.empty(0).long().to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            feats = model(inputs, return_features=True)
            total_features = torch.cat((total_features, feats))
            total_labels = torch.cat((total_labels, targets))

    ncm_classifier = KNNClassifier(args.feat_size, args.num_class)
    cfeats = get_knncentroids(feats=total_features, labels=total_labels)
    ncm_classifier.update(cfeats, device)
    ncm_logits = ncm_classifier(total_features, None)[0]

    refined_ncm_logits = ncm_logits
    refine_times = 0

    start_time = time.time()

    for _ in range(refine_times):
        mask = get_gmm_mask(refined_ncm_logits, total_labels)
        refined_cfeats = get_knncentroids(feats=total_features, labels=total_labels, mask=mask)
        ncm_classifier.update(refined_cfeats, device)
        refined_ncm_logits = ncm_classifier(total_features, None)[0]

    prob = get_gmm_prob(refined_ncm_logits, total_labels, device)

    time_elapsed = time.time() - start_time
    log_log=open('./checkpoint/%s'%(args.id)+'_log.txt','a')
    log_log.write('GMM time elapsed (seconds): %.2f\n'%(time_elapsed))
    log_log.close()
    print('GMM time elapsed (seconds): %.2f\n'%(time_elapsed))

    queue.put(prob)


# class independent loss
def eval_train(eval_loader, model, device, whichnet, queue):
    CE = nn.CrossEntropyLoss(reduction='none')
    model.eval()
    train_size = len(eval_loader.dataset)
    losses = torch.zeros(train_size)
    total_labels = torch.zeros(train_size).long()
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = CE(outputs, targets)
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]
                total_labels[index[b]]=targets[b].cpu()
    losses = (losses-losses.min())/(losses.max()-losses.min())

    input_loss = losses.reshape(-1,1)

    prob = np.zeros(train_size)
    for i in range(args.num_class):
        this_cls_idxs = (total_labels == i)
        this_cls_loss = input_loss[this_cls_idxs]
        #gmm = GaussianMixture(n_components=2, random_state=0)
        gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=1e-3,random_state=0)
        gmm.fit(this_cls_loss)
        this_cls_prob = gmm.predict_proba(this_cls_loss)
        this_cls_prob = this_cls_prob[:, gmm.means_.argmin()]
        prob[this_cls_idxs] = this_cls_prob
    queue.put(prob)


def get_gmm_mask(ncm_logits, total_labels):
    mask = torch.zeros_like(total_labels).bool()

    for i in range(args.num_class):
        this_cls_idxs = (total_labels == i)
        this_cls_logits = ncm_logits[this_cls_idxs, i].view(-1, 1).cpu().numpy()

        # normalization, note that the logits are all negative
        this_cls_logits -= np.min(this_cls_logits)
        if np.max(this_cls_logits) != 0:
            this_cls_logits /= np.max(this_cls_logits)

        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=1e-3).fit(this_cls_logits)
        gmm_preds = gmm.predict(this_cls_logits)
        inner_cluster = gmm.means_.argmax()

        this_cls_mask = mask[this_cls_idxs]
        this_cls_mask[gmm_preds == inner_cluster] = True

        if (gmm_preds != inner_cluster).all():
            this_cls_mask |= True  # not to exclude any instance

        mask[this_cls_idxs] = this_cls_mask
    return mask

def get_gmm_prob(ncm_logits, total_labels, device):
    prob = torch.zeros_like(total_labels).float()

    for i in range(args.num_class):
        this_cls_idxs = (total_labels == i)
        this_cls_logits = ncm_logits[this_cls_idxs, i].view(-1, 1).cpu().numpy()

        # normalization, note that the logits are all negative
        this_cls_logits -=  np.min(this_cls_logits)
        if np.max(this_cls_logits) != 0:
            this_cls_logits /= np.max(this_cls_logits)

        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=1e-3).fit(this_cls_logits)
        gmm_prob = gmm.predict_proba(this_cls_logits)

        this_cls_prob = torch.Tensor(gmm_prob[:, gmm.means_.argmax()]).to(device)
        prob[this_cls_idxs] = this_cls_prob

    return prob.cpu().numpy()

def get_knncentroids(feats=None, labels=None, mask=None):

    feats = feats.cpu().numpy()
    labels = labels.cpu().numpy()

    featmean = feats.mean(axis=0)

    def get_centroids(feats_, labels_, mask_=None):
        if mask_ is None:
            mask_ = np.ones_like(labels_).astype('bool')
        elif isinstance(mask_, torch.Tensor):
            mask_ = mask_.cpu().numpy()

        centroids = []
        for i in np.unique(labels_):
            centroids.append(np.mean(feats_[(labels_==i) & mask_], axis=0))
        return np.stack(centroids)

    # Get unnormalized centorids
    un_centers = get_centroids(feats, labels, mask)

    # Get l2n centorids
    l2n_feats = torch.Tensor(feats.copy())
    norm_l2n = torch.norm(l2n_feats, 2, 1, keepdim=True)
    l2n_feats = l2n_feats / norm_l2n
    l2n_centers = get_centroids(l2n_feats.numpy(), labels, mask)

    # Get cl2n centorids
    cl2n_feats = torch.Tensor(feats.copy())
    cl2n_feats = cl2n_feats - torch.Tensor(featmean)
    norm_cl2n = torch.norm(cl2n_feats, 2, 1, keepdim=True)
    cl2n_feats = cl2n_feats / norm_cl2n
    cl2n_centers = get_centroids(cl2n_feats.numpy(), labels, mask)

    return {'mean': featmean,
            'uncs': un_centers,
            'l2ncs': l2n_centers,
            'cl2ncs': cl2n_centers}


def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model(device):
    model = InceptionResNetV2(num_classes=args.num_class)
    model = model.to(device)
    return model

if __name__ == "__main__":

    mp.set_start_method('spawn')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.baseline:
        if args.cls_ind_dist:
            args.id += '_cls_ind_dist'
        elif args.cls_ind_loss:
            args.id += '_cls_ind_loss'

        #if args.resume >= 0:
        #    args.id += f'_resume_from_e{args.resume}'
    else:
        args.id += '_baseline'

    if args.drw:
        args.id += '_drw'

    if args.imb_ratio < 0.5:
        args.id += f'_imb_ratio_{args.imb_ratio}'

    args.id += f'_warmup_{args.warm_up}'

    stats_log=open('./checkpoint/%s'%(args.id)+'_stats.txt','w')
    test_log=open('./checkpoint/%s'%(args.id)+'_acc.txt','w')

    start_epoch = 0
    warm_up = args.warm_up

    loader = dataloader.webvision_dataloader(batch_size=args.batch_size,num_class = args.num_class,num_workers=8,root_dir=args.data_path,log=stats_log, imb_ratio=args.imb_ratio)

    print('| Building net')

    net1 = create_model(cuda1)
    net2 = create_model(cuda2)

    net1_clone = create_model(cuda2)
    net2_clone = create_model(cuda1)

    cudnn.benchmark = True

    optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    if args.resume >= 0:
        if len(args.resume_id) > 0:
            ckpt_file = f'./checkpoint/model/{args.resume_id}'
        else:
            ckpt_file = f'./checkpoint/model/{args.id}_e{args.resume}.pth'
        print('| Loading warmup model of from', ckpt_file)
        checkpoint = torch.load(ckpt_file)
        net1.load_state_dict(checkpoint['net1'])
        net1.to(cuda1)
        net2.load_state_dict(checkpoint['net2'])
        net2.to(cuda2)
        net1_clone.load_state_dict(checkpoint['net1_clone'])
        net2_clone.load_state_dict(checkpoint['net2_clone'])
        optimizer1.load_state_dict(checkpoint['optimizer1'])
        optimizer2.load_state_dict(checkpoint['optimizer2'])
        prob1 = checkpoint['prob1']
        prob2 = checkpoint['prob2']
        start_epoch = args.resume + 1

    #conf_penalty = NegEntropy()
    web_valloader = loader.run('test')
    imagenet_valloader = loader.run('imagenet')

    for epoch in range(start_epoch, args.num_epochs+1):
        lr=args.lr
        if epoch >= 50:
            lr /= 10
        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr

        if epoch<warm_up:
            warmup_trainloader1 = loader.run('warmup')
            warmup_trainloader2 = loader.run('warmup')
            p1 = mp.Process(target=warmup, args=(epoch,net1,optimizer1,warmup_trainloader1,cuda1,'net1'))
            p2 = mp.Process(target=warmup, args=(epoch,net2,optimizer2,warmup_trainloader2,cuda2,'net2'))
            p1.start()
            p2.start()

        else:
            pred1 = (prob1 > args.p_threshold)
            pred2 = (prob2 > args.p_threshold)

            labeled_trainloader1, unlabeled_trainloader1 = loader.run('train',pred2,prob2) # co-divide
            labeled_trainloader2, unlabeled_trainloader2 = loader.run('train',pred1,prob1) # co-divide

            p1 = mp.Process(target=train, args=(epoch,net1,net2_clone,optimizer1,labeled_trainloader1, unlabeled_trainloader1,cuda1,'net1'))
            p2 = mp.Process(target=train, args=(epoch,net2,net1_clone,optimizer2,labeled_trainloader2, unlabeled_trainloader2,cuda2,'net2'))
            p1.start()
            p2.start()

        p1.join()
        p2.join()

        net1_clone.load_state_dict(net1.state_dict())
        net2_clone.load_state_dict(net2.state_dict())

        q1 = mp.Queue()
        q2 = mp.Queue()
        p1 = mp.Process(target=test, args=(epoch,net1,net2_clone,web_valloader,cuda1,q1))
        p2 = mp.Process(target=test, args=(epoch,net1_clone,net2,imagenet_valloader,cuda2,q2))

        p1.start()
        p2.start()

        web_acc = q1.get()
        imagenet_acc = q2.get()

        p1.join()
        p2.join()

        print("\n| Test Epoch #%d\t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%)\n"%(epoch,web_acc[0],web_acc[1],imagenet_acc[0],imagenet_acc[1]))
        test_log.write('Epoch:%d \t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%)\n'%(epoch,web_acc[0],web_acc[1],imagenet_acc[0],imagenet_acc[1]))
        test_log.flush()

        eval_loader1 = loader.run('eval_train')
        eval_loader2 = loader.run('eval_train')
        q1 = mp.Queue()
        q2 = mp.Queue()

        if args.cls_ind_dist:
            p1 = mp.Process(target=eval_train_rolt, args=(eval_loader1,net1,cuda1,'net1',q1))
            p2 = mp.Process(target=eval_train_rolt, args=(eval_loader2,net2,cuda2,'net2',q2))
        elif args.cls_ind_loss:
            p1 = mp.Process(target=eval_train, args=(eval_loader1,net1,cuda1,'net1',q1))
            p2 = mp.Process(target=eval_train, args=(eval_loader2,net2,cuda2,'net2',q2))
        else:
            p1 = mp.Process(target=eval_train_baseline, args=(eval_loader1,net1,cuda1,'net1',q1))
            p2 = mp.Process(target=eval_train_baseline, args=(eval_loader2,net2,cuda2,'net2',q2))

        p1.start()
        p2.start()

        prob1 = q1.get()
        prob2 = q2.get()

        p1.join()
        p2.join()

        if epoch == warm_up - 1 or epoch % 40 == 0:
            torch.save({
                'net1': net1.state_dict(),
                'net2': net2.state_dict(),
                'net1_clone': net1_clone.state_dict(),
                'net2_clone': net2_clone.state_dict(),
                'optimizer1': optimizer1.state_dict(),
                'optimizer2': optimizer2.state_dict(),
                'prob1': prob1,
                'prob2': prob2,
            }, f'./checkpoint/model/{args.id}_e{epoch}.pth')
