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
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
# parser.add_argument('--data_path', default='./cifar-10', type=str, help='path to dataset')
parser.add_argument('--data_path', default='./data', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--imb_type', default='exp', type=str)
parser.add_argument('--imb_factor', default=0.01, type=float)
parser.add_argument('--arch', default='resnet18', type=str, help='resnet18/32')
parser.add_argument('--baseline', action='store_true')
parser.add_argument('--cls_ind', action='store_true')
parser.add_argument('--recycle', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--drw', action='store_true')
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

resume_epoch = 200
drw_epoch = 200
recycle_epoch = 200

# Training
def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader,current_labels):
    net.train()
    net2.eval() #fix one network and train the other
    
    cls_num_list = np.array([  (current_labels == i).sum() for i in range(args.num_class)  ])
    # cls_num_list = np.array([  (noisy_labels == i).sum() for i in range(args.num_class)  ])
    idx = epoch // drw_epoch
    betas = [0, 0.9999]
    effective_num = 1.0 - np.power(betas[idx], cls_num_list)
    per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()

    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x, index_x) in enumerate(labeled_trainloader):    
        labels_x = current_labels[index_x]
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()                 
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

        all_labels = current_labels[index_x].repeat(2)
        all_weights = per_cls_weights[all_labels]

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        weight_a, weight_b = all_weights, all_weights[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
        mixed_weight = l * weight_a + (1 - l) * weight_b
                
        logits = net(mixed_input)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]        
        
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up, mixed_weight)
        
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
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
        sys.stdout.flush()

def warmup(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
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
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
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

    total_preds = torch.cat(total_preds)
    total_targets = torch.cat(total_targets)
    cls_acc = [ round( 100. * ((total_preds == total_targets) & (total_targets == i)).sum().item() / (total_targets == i).sum().item(), 2) \
                for i in range(args.num_class)]

    print("\n| Test Epoch #%d\t Accuracy: %.2f%% %s\n" %(epoch,acc, str(cls_acc)))
    test_log.write('Epoch:%d   Accuracy:%.2f %s\n'%(epoch,acc, str(cls_acc)))
    test_log.flush()  

def eval_train_loss(model,all_loss):    
    model.eval()
    # losses = torch.zeros(50000)    
    losses = torch.zeros(train_size)
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]         
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)

    if args.r==0.9: # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1,1)
    else:
        input_loss = losses.reshape(-1,1)
    
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]         
    return prob,all_loss

def eval_train_cls_ind_loss(model,all_loss):    
    model.eval()
    # losses = torch.zeros(50000)    
    losses = torch.zeros(train_size)
    total_labels = torch.zeros(train_size).long()
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]       
                total_labels[index[b]]=targets[b]  
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)

    if args.r==0.9: # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1,1)
    else:
        input_loss = losses.reshape(-1,1)
    
    # fit a two-component GMM to the loss
    # gmm = GaussianMixture(n_components=2, random_state=0)
    # gmm.fit(input_loss)
    # prob = gmm.predict_proba(input_loss) 
    # prob = prob[:,gmm.means_.argmin()]         
    # return prob,all_loss

    prob = np.zeros(train_size)
    for i in range(args.num_class):
        this_cls_idxs = (total_labels == i)
        this_cls_loss = input_loss[this_cls_idxs]
        gmm = GaussianMixture(n_components=2, random_state=0)
        gmm.fit(this_cls_loss)
        this_cls_prob = gmm.predict_proba(this_cls_loss)
        this_cls_prob = this_cls_prob[:, gmm.means_.argmin()]
        prob[this_cls_idxs] = this_cls_prob
    return prob,all_loss


def eval_train_cls_ind_dist(model,all_loss=None):    
    model.eval()

    total_features = torch.empty((0, feat_size)).cuda()
    total_labels = torch.empty(0).long().cuda()

    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            feats = model(inputs, return_features=True)
            total_features = torch.cat((total_features, feats))
            total_labels = torch.cat((total_labels, targets))

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

    prob = get_gmm_prob(refined_ncm_logits, total_labels)
    return prob, None

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


def eval_train_recycle(model,all_loss=None):    
    model.eval()

    total_features = torch.empty((0, feat_size)).cuda()
    total_labels = torch.empty(0).long().cuda()

    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            feats = model(inputs, return_features=True)
            total_features = torch.cat((total_features, feats))
            total_labels = torch.cat((total_labels, targets))

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

    prob = torch.zeros_like(refined_ncm_logits).float()
    prob_self = torch.zeros_like(total_labels).float()
    all_prob = torch.zeros_like(total_labels).float()
    prob2 = torch.zeros_like(refined_ncm_logits).float()
    avg_fix_prob = torch.zeros(args.num_class).float().cuda()

    this_cls_covs = np.zeros(args.num_class)

    for i in range(args.num_class):
        this_cls_idxs = (total_labels == i)
        this_cls_logits = refined_ncm_logits[this_cls_idxs, i].view(-1, 1).cpu().numpy()
        
        # normalization, note that the logits are all negative
        normed_ncm_logits = refined_ncm_logits.cpu().numpy()
        normed_ncm_logits -= np.min(this_cls_logits)
        if np.max(this_cls_logits) - np.min(this_cls_logits) != 0:
            normed_ncm_logits /= np.max(this_cls_logits) - np.min(this_cls_logits)
        
        # normalization, note that the logits are all negative
        this_cls_logits -= np.min(this_cls_logits)
        if np.max(this_cls_logits) != 0:
            this_cls_logits /= np.max(this_cls_logits)

        gmm = GaussianMixture(n_components=2, random_state=0).fit(this_cls_logits)
        this_cls_covs[i] = gmm.covariances_[gmm.means_.argmax()]

        gmm_prob = gmm.predict_proba(normed_ncm_logits[:, i].reshape(-1, 1))
        prob_i = torch.Tensor(gmm_prob[:, gmm.means_.argmax()]).cuda()
        prob[:,i] = prob_i
        prob_self[this_cls_idxs] = prob_i[this_cls_idxs]

        this_cls_fix_idxs = this_cls_idxs & (prob_i > 0.5)
        all_prob = np.exp(gmm._estimate_weighted_log_prob(normed_ncm_logits[:, i].reshape(-1, 1)))
        prob2_i = torch.Tensor(all_prob[:, gmm.means_.argmax()]).cuda()

        prob2[:,i] = prob2_i
        avg_fix_prob[i] = prob2_i[this_cls_fix_idxs].mean()

    # cls_covs_mean = this_cls_covs.mean()
    # unrel_cls = this_cls_covs > cls_covs_mean
    # avg_fix_prob[unrel_cls] = float('inf')

    max_prob, _ = prob.max(dim=1)
    max_prob2, current_labels = prob2.max(dim=1)


    fix_idxs = prob_self > 0.5

    max_prob[fix_idxs], current_labels[fix_idxs] = prob_self[fix_idxs], total_labels[fix_idxs]

    per_sample_recycle_thresh = avg_fix_prob[current_labels]

    # free_idxs = (~fix_idxs) & (max_prob < avg_fix_prob)
    free_idxs = (~fix_idxs) & (max_prob2 < per_sample_recycle_thresh)

    recycle_idxs = ~(fix_idxs | free_idxs)
    max_prob[free_idxs], current_labels[free_idxs] = 0, -1
    max_prob[recycle_idxs] *= 1.0

    for i in range(len(current_labels)):
        if fix_idxs[i]:
            continue
        
        current_labels[i] = -1
        max_prob[i] = 0
        recycle_idxs[i] = False

        if clean_labels[i] < 50:
            current_labels[i] = clean_labels[i]
            max_prob[i] = 1
            recycle_idxs[i] = True


    # for i in range(args.num_class):
    #     this_cls_fix_idxs = fix_idxs & (current_labels == i)
    #     this_cls_recycle_idxs = recycle_idxs & (current_labels == i)
        
    #     # if sum(this_cls_recycle_idxs) > sum(this_cls_fix_idxs) // 1:
    #     #     this_cls_prob2 = max_prob2[this_cls_recycle_idxs]
    #     #     idx = np.argsort(this_cls_prob2.cpu().numpy())[0:sum(this_cls_recycle_idxs) - sum(this_cls_fix_idxs) // 1]
    #     #     current_labels[this_cls_recycle_idxs.nonzero().view(-1)[idx]] = -1
    #     #     max_prob[this_cls_recycle_idxs.nonzero().view(-1)[idx]] = 0
        
    #     this_cls_clean_idxs = recycle_idxs & (clean_labels == i)

    #     if i < 50:
    #         current_labels[this_cls_recycle_idxs] = -1
    #         max_prob[this_cls_recycle_idxs] = 0
    #     else:
    #         current_labels[this_cls_recycle_idxs] = clean_labels[this_cls_recycle_idxs]
    #         max_prob[this_cls_recycle_idxs] = 1


    recycle_idxs[current_labels == -1] = False

    for idxs, tag in zip( (fix_idxs, recycle_idxs), ('fixed', 'recycled') ):
        pred = current_labels[idxs]
        target = clean_labels[idxs]
        print(tag, idxs.sum().item())
        test_log.write(tag + ' ' + str(idxs.sum().item()) + '\n')
        for i in range(args.num_class):
            # if(this_cls_covs[i] > cls_covs_mean):
            #     print('!!!', end='')

            p_idxs = pred == i
            r_idxs = target == i
            p = (pred[p_idxs] == target[p_idxs]).float().mean() if p_idxs.any() else 0.0
            r = (pred[r_idxs] == target[r_idxs]).float().sum() / (clean_labels == i).float().sum()
            print('%d:(%.4f %.4f)' % (i, p, r), end=' ')
            test_log.write('%d:(%.4f %.4f) ' % (i, p, r))

            # if tag == 'recycled' and p <= 1:
            #     print()
            #     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            #     print(i)
            #     this_cls_idxs = (total_labels == i)
            #     this_cls_fix_idxs = this_cls_idxs & (prob[:,i] > 0.5)
            #     # print(sorted(prob2[this_cls_fix_idxs,i].cpu().numpy()))
            #     # print( prob2[this_cls_fix_idxs,i].cpu().numpy().mean() )
            #     # print(sorted(refined_ncm_logits[this_cls_fix_idxs, i].cpu().numpy()))
            #     print()
        print()
        test_log.write('\n')
    test_log.flush()  

    return max_prob.cpu().numpy(), None, current_labels.cpu().long()
    # return prob_self.cpu().numpy(), None, total_labels.cpu()


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
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up, weight=None):
        probs_u = torch.softmax(outputs_u, dim=1)
        # weight: N*1 tensor
        if args.drw is False:
            Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        else:
            Lx = -torch.sum(weight * torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1)) / torch.sum(weight)
        # Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
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

# stats_log=open('./checkpoint/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_stats.txt','w') 
# test_log=open('./checkpoint/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_acc.txt','w')     
store_name = '_'.join([args.dataset, args.arch, args.imb_type, str(args.imb_factor), args.noise_mode, str(args.r)])
if args.baseline:
    store_name += '_baseline'
elif args.cls_ind:
    store_name += '_cls_ind'
elif args.recycle:
    store_name += '_recycle'

if args.drw:
    store_name += '_drw'
if args.resume:
    store_name += '_resume'

stats_log=open('./checkpoint/%s'%(store_name)+'_stats.txt','w') 
test_log=open('./checkpoint/%s'%(store_name)+'_acc.txt','w')    

start_epoch = 0

if args.dataset=='cifar10':
    warm_up = 10
elif args.dataset=='cifar100':
    warm_up = 30

# loader = dataloader.cifar_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=5,\
#     root_dir=args.data_path,log=stats_log,noise_file='%s/%.1f_%s.json'%(args.data_path,args.r,args.noise_mode))
loader = dataloader.cifar_dataloader(args.dataset, imb_type=args.imb_type, imb_factor=args.imb_factor, noise_mode=args.noise_mode, r=args.r,\
    batch_size=args.batch_size, num_workers=5, root_dir=args.data_path, log=stats_log)
train_size = len(loader.run('warmup').dataset)
args.num_class = 100 if args.dataset == 'cifar100' else 10
feat_size = 512 if args.arch == 'resnet18' else 64
ncm_classifier = KNNClassifier(feat_size, args.num_class)
# current_labels1 = loader.run('warmup').dataset.noise_label
# current_labels2 = loader.run('warmup').dataset.noise_label
noisy_labels = torch.Tensor(loader.run('warmup').dataset.noise_label).long()
clean_labels = torch.Tensor(loader.run('warmup').dataset.clean_targets).long().cuda()

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


if args.resume:
    print(f'| Loading model from {resume_epoch} epoch.')
    checkpoint = torch.load(f'./checkpoint/model_{store_name}'.split('_resume')[0])
    net1.load_state_dict(checkpoint['net1'])
    net2.load_state_dict(checkpoint['net2'])
    optimizer1.load_state_dict(checkpoint['optimizer1'])
    optimizer2.load_state_dict(checkpoint['optimizer2'])
    prob1 = checkpoint['prob1']
    prob2 = checkpoint['prob2']
    start_epoch = resume_epoch + 1


CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
if args.noise_mode=='asym':
    conf_penalty = NegEntropy()

all_loss = [[],[]] # save the history of losses from two networks

for epoch in range(start_epoch, args.num_epochs+1):   
    lr=args.lr
    if epoch >= 150:
        lr /= 10      
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr       
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr          
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')   
    
    if epoch<warm_up:       
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(epoch,net1,optimizer1,warmup_trainloader)    
        print('\nWarmup Net2')
        warmup(epoch,net2,optimizer2,warmup_trainloader) 
   
    else:         
        current_labels1 = noisy_labels
        current_labels2 = noisy_labels

        if args.baseline:
            prob1,all_loss[0]=eval_train_baseline(net1,all_loss[0])
            prob2,all_loss[1]=eval_train_baseline(net2,all_loss[1])
        elif args.cls_ind:
            prob1,all_loss[0]=eval_train_cls_ind(net1,all_loss[0])
            prob2,all_loss[1]=eval_train_cls_ind(net2,all_loss[1])
        elif args.recycle and epoch > recycle_epoch:
            prob1,all_loss[0],current_labels1=eval_train_recycle(net1,all_loss[0])
            prob2,all_loss[1],current_labels2=eval_train_recycle(net2,all_loss[1])
        else:
            prob1,all_loss[0]=eval_train(net1,all_loss[0])   
            prob2,all_loss[1]=eval_train(net2,all_loss[1])          

        pred1 = (prob1 > args.p_threshold)      
        pred2 = (prob2 > args.p_threshold)      

        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2) # co-divide
        train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader, current_labels2) # train net1  
        
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1) # co-divide
        train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader, current_labels1) # train net2         

    test(epoch,net1,net2)  


    if epoch == resume_epoch:
        torch.save({
            'net1': net1.state_dict(),
            'net2': net2.state_dict(),
            'optimizer1': optimizer1.state_dict(),
            'optimizer2': optimizer2.state_dict(),
            'prob1': prob1,
            'prob2': prob2,
        }, f'./checkpoint/model_{store_name}')
        print('| Done saving model.')