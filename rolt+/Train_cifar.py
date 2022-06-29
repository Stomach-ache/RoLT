from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from resnet_cifar import *
from KNNClassifier import KNNClassifier
from sklearn.mixture import GaussianMixture
import dataloader_cifar as dataloader

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123, type=int)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
# parser.add_argument('--data_path', default='./cifar-10', type=str, help='path to dataset')
parser.add_argument('--data_path', default='./data', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--imb_type', default='exp', type=str)
parser.add_argument('--imb_factor', default=0.01, type=float)
parser.add_argument('--noise_mode',  default='imb')
parser.add_argument('--noise_ratio', default=0.5, type=float, help='noise ratio')
parser.add_argument('--arch', default='resnet18', type=str, help='resnet18/32')
parser.add_argument('--cls_ind', action='store_true')
parser.add_argument('-b', '--basis', default='loss', type=str, help='loss/dist')
parser.add_argument('-w', '--warm_up', default=None, type=int)
parser.add_argument('-d', '--drw', default=None, type=int)
parser.add_argument('-r', '--resume', default=None, type=int)
args = parser.parse_args()

def set_seed():
    seed = args.seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.cuda.set_device(args.gpuid)
set_seed()


# Training
def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader,current_labels):
    net.train()
    net2.eval() #fix one network and train the other

    if args.drw is not None:
        cls_num_list = np.array([  (current_labels == i).sum() for i in range(args.num_class)  ])
        # cls_num_list = np.array([  (noisy_labels == i).sum() for i in range(args.num_class)  ])
        idx = (epoch - 1) // args.drw
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], cls_num_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
    else:
        per_cls_weights = None

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x, index_x) in enumerate(labeled_trainloader):
        labels_x = current_labels[index_x]
        try:
            inputs_u, inputs_u2, index_u = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, index_u = unlabeled_train_iter.next()
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)
        w_x = w_x.view(-1,1).type(torch.FloatTensor)

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

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

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        logits = net(mixed_input)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]

        if args.drw and per_cls_weights is not None:
            labels_x, labels_u = current_labels[index_x], current_labels[index_u]

            all_labels = torch.cat([labels_x, labels_x, labels_u, labels_u], dim=0)
            all_weights = per_cls_weights[all_labels]

            weight_a, weight_b = all_weights, all_weights[idx]
            mixed_weight = weight_a
            #mixed_weight = l * weight_a + (1 - l) * weight_b

            weight_x = mixed_weight[:batch_size*2]
            weight_u = None
        else:
            weight_x = None
            weight_u = None

        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up, weight_x, weight_u)

        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + lamb * Lu  + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%s_%g_%s_%g | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                %(args.dataset, args.imb_type, args.imb_factor, args.noise_mode, args.noise_ratio, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
        sys.stdout.flush()

def warmup(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, index) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = CEloss(outputs, labels)
        if args.noise_mode=='asym':  # penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty
        elif args.noise_mode=='sym':
            L = loss
        elif args.noise_mode=='imb':
            L = loss
        L.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%s_%g_%s_%g | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.imb_type, args.imb_factor, args.noise_mode, args.noise_ratio, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()

def test(epoch,net1,net2):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    total_preds = []
    total_targets = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()

            total_preds.append(predicted)
            total_targets.append(targets)

    acc = 100.*correct/total
    global best_overall_acc, best_overall_acc_epoch
    if acc > best_overall_acc:
        best_overall_acc = acc
        best_overall_acc_epoch = epoch

    total_preds = torch.cat(total_preds)
    total_targets = torch.cat(total_targets)
    cls_acc = [ round( 100. * ((total_preds == total_targets) & (total_targets == i)).sum().item() / (total_targets == i).sum().item(), 2) \
                for i in range(args.num_class)]

    print("\n| Test Epoch #%d\t Accuracy: %.2f%% %s\n" %(epoch,acc, str(cls_acc)))
    test_log.write('Epoch:%d   Accuracy:%.2f %s\n'%(epoch,acc, str(cls_acc)))
    print("\n| Test Epoch #%d\t Best Overall Accuracy: %.2f%%\n" %(best_overall_acc_epoch,best_overall_acc))
    test_log.write('Epoch:%d  Best Overall Accuracy:%.2f\n'%(best_overall_acc_epoch,best_overall_acc))
    test_log.flush()

def eval_train(model,all_loss):
    model.eval()

    total_features = torch.zeros((train_size, feat_size))
    total_labels = torch.zeros(train_size).long()
    losses = torch.zeros(train_size)

    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            feats, outputs = model(inputs, return_features=True)
            loss = CE(outputs, targets)
            for b in range(inputs.size(0)):
                total_features[index[b]] = feats[b]
                total_labels[index[b]] = targets[b]
                losses[index[b]] = loss[b]

    if args.basis == 'loss':
        losses = (losses-losses.min()) / (losses.max()-losses.min())
        all_loss.append(losses)

        if args.noise_ratio==0.9: # average loss over last 5 epochs to improve convergence stability
            history = torch.stack(all_loss)
            input_loss = history[-5:].mean(0)
            input_loss = input_loss.reshape(-1,1)
        else:
            input_loss = losses.reshape(-1,1)

        if not args.cls_ind:
            # fit a two-component GMM to the loss
            gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
            gmm.fit(input_loss)
            prob = gmm.predict_proba(input_loss)
            prob = prob[:,gmm.means_.argmin()]
        else:
            prob = np.zeros(train_size)
            for i in range(args.num_class):
                this_cls_idxs = (total_labels == i)
                this_cls_loss = input_loss[this_cls_idxs]
                gmm = GaussianMixture(n_components=2, random_state=0)
                gmm.fit(this_cls_loss)
                this_cls_prob = gmm.predict_proba(this_cls_loss)
                this_cls_prob = this_cls_prob[:, gmm.means_.argmin()]
                prob[this_cls_idxs] = this_cls_prob

    elif args.basis == 'dist':
        total_features = total_features.cuda()
        total_labels = total_labels.cuda()

        cfeats = get_knncentroids(feats=total_features, labels=total_labels)
        ncm_classifier.update(cfeats)
        ncm_logits = ncm_classifier(total_features, None)[0]

        refined_ncm_logits = ncm_logits
        refine_times = 1
        for _ in range(refine_times):
            mask = get_gmm_mask(refined_ncm_logits, total_labels)
            refined_cfeats = get_knncentroids(feats=total_features, labels=total_labels, mask=mask)
            ncm_classifier.update(refined_cfeats)
            refined_ncm_logits = ncm_classifier(total_features, None)[0]

        if args.cls_ind:
            prob = get_gmm_prob(refined_ncm_logits, total_labels)
        else:
            input_logits = ncm_logits[total_labels>=0, total_labels].view(-1, 1).cpu().numpy()
            input_logits = (input_logits - input_logits.min()) / (input_logits.max() - input_logits.min())

            gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
            gmm.fit(input_logits)
            gmm_prob = gmm.predict_proba(input_logits)
            prob = gmm_prob[:, gmm.means_.argmax()]

    return prob, all_loss

def get_gmm_mask(ncm_logits, total_labels):
    mask = torch.zeros_like(total_labels).bool()

    for i in range(args.num_class):
        this_cls_idxs = (total_labels == i)
        this_cls_logits = ncm_logits[this_cls_idxs, i].view(-1, 1).cpu().numpy()

        # normalization, note that the logits are all negative
        this_cls_logits -= np.min(this_cls_logits)
        if np.max(this_cls_logits) != 0:
            this_cls_logits /= np.max(this_cls_logits)

        gmm = GaussianMixture(n_components=2, random_state=0).fit(this_cls_logits)
        gmm_preds = gmm.predict(this_cls_logits)
        inner_cluster = gmm.means_.argmax()

        this_cls_mask = mask[this_cls_idxs]
        this_cls_mask[gmm_preds == inner_cluster] = True

        if (gmm_preds != inner_cluster).all():
            this_cls_mask |= True  # not to exclude any instance

        mask[this_cls_idxs] = this_cls_mask
    return mask

def get_gmm_prob(ncm_logits, total_labels):
    prob = torch.zeros_like(total_labels).float()

    for i in range(args.num_class):
        this_cls_idxs = (total_labels == i)
        this_cls_logits = ncm_logits[this_cls_idxs, i].view(-1, 1).cpu().numpy()

        # normalization, note that the logits are all negative
        this_cls_logits -= np.min(this_cls_logits)
        if np.max(this_cls_logits) != 0:
            this_cls_logits /= np.max(this_cls_logits)

        gmm = GaussianMixture(n_components=2, random_state=0).fit(this_cls_logits)
        gmm_prob = gmm.predict_proba(this_cls_logits)
        this_cls_prob = torch.Tensor(gmm_prob[:, gmm.means_.argmax()]).cuda()
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
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up, weight_x=None, weight_u=None):
        probs_u = torch.softmax(outputs_u, dim=1)

        # Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        # Lu = torch.mean((probs_u - targets_u)**2)

        # weight: N*1 tensor
        if weight_x is not None:
            Lx = -torch.sum(weight_x * torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1)) / torch.sum(weight_x)
        else:
            Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))

        if weight_u is not None:
            Lu = torch.sum(weight_u * torch.mean((probs_u - targets_u)**2, dim=1)) / torch.sum(weight_u)
        else:
            Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    if args.arch == 'resnet18':
        model = ResNet18(num_classes=args.num_class)
    else:
        model = resnet32(num_classes=args.num_class)
    model = model.cuda()
    return model

if args.warm_up is not None:
    warm_up = args.warm_up
else:
    if args.imb_factor == 1:
        if args.dataset=='cifar10':
            warm_up = 10
        elif args.dataset=='cifar100':
            warm_up = 30
    else:
        warm_up = 50

store_name = '_'.join([args.dataset, args.arch, args.imb_type, str(args.imb_factor), args.noise_mode, str(args.noise_ratio)])

log_name = store_name + f'_w{warm_up}'
if args.cls_ind:
    log_name += '_cls_ind'
log_name += f'_{args.basis}'

if args.drw is not None:
    log_name += f'_d{args.drw}'
if args.resume is not None:
    log_name += f'_r{args.resume}'

log_name += f'_{args.seed}'

stats_log = open('./checkpoint/log_%s'%(log_name)+'_stats.txt','w')
test_log = open('./checkpoint/log_%s'%(log_name)+'_acc.txt','w')

loader = dataloader.cifar_dataloader(args.dataset, imb_type=args.imb_type, imb_factor=args.imb_factor, noise_mode=args.noise_mode, noise_ratio=args.noise_ratio,\
    batch_size=args.batch_size, num_workers=5, root_dir=args.data_path, log=stats_log, seed=args.seed)
train_size = len(loader.run('warmup').dataset)
args.num_class = 100 if args.dataset == 'cifar100' else 10
feat_size = 512 if args.arch == 'resnet18' else 64
ncm_classifier = KNNClassifier(feat_size, args.num_class)
current_labels1 = loader.run('warmup').dataset.noise_label
current_labels2 = loader.run('warmup').dataset.noise_label
noisy_labels = torch.Tensor(loader.run('warmup').dataset.noise_label).long()
clean_labels = torch.Tensor(loader.run('warmup').dataset.clean_targets).long().cuda()

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

model_name = store_name
if args.resume is not None:
    if args.resume > warm_up:
        model_name += f'_w{warm_up}'
    if args.drw is not None and args.resume > args.drw:
        model_name += f'_d{args.drw}'

    resume_path = f'./checkpoint/model_{model_name}_e{args.resume}.pth'
    print(f'| Loading model from {resume_path}')
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path)
        net1.load_state_dict(ckpt['net1'])
        net2.load_state_dict(ckpt['net2'])
        optimizer1.load_state_dict(ckpt['optimizer1'])
        optimizer2.load_state_dict(ckpt['optimizer2'])
        prob1 = ckpt['prob1']
        prob2 = ckpt['prob2']
        start_epoch = args.resume + 1
    else:
        print('| Failed to resume.')
        model_name = store_name
        start_epoch = 1
else:
    start_epoch = 1

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
if args.noise_mode=='asym':
    conf_penalty = NegEntropy()

all_loss = [[],[]] # save the history of losses from two networks

best_overall_acc, best_overall_acc_epoch = 0, 0
print (best_overall_acc)

for epoch in range(start_epoch, args.num_epochs + 1):
    lr = args.lr
    if epoch > 150:
        lr /= 10
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')

    if epoch <= warm_up:
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(epoch,net1,optimizer1,warmup_trainloader)
        print('\nWarmup Net2')
        warmup(epoch,net2,optimizer2,warmup_trainloader)

    else:
        current_labels1 = noisy_labels
        current_labels2 = noisy_labels

        prob1, all_loss[0] = eval_train(net1, all_loss[0])
        prob2, all_loss[1] = eval_train(net2, all_loss[1])

        pred1 = (prob1 > args.p_threshold)
        pred2 = (prob2 > args.p_threshold)

        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2) # co-divide
        train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader, current_labels2) # train net1

        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1) # co-divide
        train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader, current_labels1) # train net2

    test(epoch,net1,net2)

    if epoch in [warm_up, args.drw]:
        save_path = f'./checkpoint/model_{model_name}_e{epoch}.pth'
        print(f'| Saving model to {save_path}')

        ckpt = {'net1': net1.state_dict(),
                'net2': net2.state_dict(),
                'optimizer1': optimizer1.state_dict(),
                'optimizer2': optimizer2.state_dict(),
                'prob1': prob1 if 'prob1' in dir() else None,
                'prob2': prob2 if 'prob2' in dir() else None}
        torch.save(ckpt, save_path)

    if epoch == warm_up:
        model_name += f'_w{warm_up}'
        if args.cls_ind:
            model_name += '_cls_ind'
        model_name += f'_{args.basis}'
    if args.drw is not None and epoch == args.drw:
        model_name += f'_d{args.drw}'
