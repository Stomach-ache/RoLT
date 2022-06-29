"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""

import os
import copy
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
from logger import Logger
import time
import numpy as np
import warnings
import pdb
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from models.KNNClassifier import KNNClassifier

import seaborn as sns
sns.set(style='darkgrid')
sns.color_palette('bright')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rc('font', size=6)

def distribution_calibration(query, base_means, base_cov, k,alpha=0.21):
    dist = []
    for i in range(len(base_means)):
        dist.append(np.linalg.norm(query-base_means[i]))
    index = np.argpartition(dist, k)[:k]
    mean = np.concatenate([np.array(base_means)[index], query[np.newaxis, :]])
    calibrated_mean = np.mean(mean, axis=0)
    calibrated_cov = np.mean(np.array(base_cov)[index], axis=0)+alpha

    return calibrated_mean, calibrated_cov

class model ():
    
    def __init__(self, config, data, test=False):
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.training_opt = self.config['training_opt']
        self.memory = self.config['memory']
        self.data = data
        self.test_mode = test
        self.num_gpus = torch.cuda.device_count()
        self.do_shuffle = config['shuffle'] if 'shuffle' in config else False
        self.do_cleaning = config['cleaning']
        self.last_epoch = 0

        # Setup logger
        self.logger = Logger(self.training_opt['log_dir'])
        
        # Initialize model
        self.init_models()

        self.ncm_classifier = KNNClassifier(self.training_opt['feature_dim'], self.training_opt['num_classes'])

        # Load pre-trained model parameters
        if 'model_dir' in self.config and self.config['model_dir'] is not None:
            self.load_model(self.config['model_dir'])

        # Under training mode, initialize training steps, optimizers, schedulers, criterions, and centroids
        if not self.test_mode:

            # If using steps for training, we need to calculate training steps 
            # for each epoch based on actual number of training data instead of 
            # oversampled data number 
            print('Using steps for training.')
            self.training_data_num = len(self.data['train'].dataset)
            self.epoch_steps = int(self.training_data_num  \
                                   / self.training_opt['batch_size'])

            # Initialize model optimizer and scheduler
            print('Initializing model optimizer.')
            self.scheduler_params = self.training_opt['scheduler_params']
            self.model_optimizer, \
            self.model_optimizer_scheduler = self.init_optimizers(self.model_optim_params_list)
            self.init_criterions()
            if self.memory['init_centroids']:
                self.criterions['FeatureLoss'].centroids.data = \
                    self.centroids_cal(self.data['train_plain'])
            
            # Load last-trained model parameters
            self.load_last = config['load_last']
            if self.load_last:
                self.load_last_model()
            
            self.load_last_v2 = config['load_last_v2']
            if self.load_last_v2:
                self.load_last_model('last_v2')

            # Set up log file
            if config['tag'] is not None:
                tag = config['tag']
                self.log_file = os.path.join(self.training_opt['log_dir'], f'log_{tag}.txt')
            elif self.load_last:
                self.log_file = os.path.join(self.training_opt['log_dir'], 'log_new.txt')
            elif self.load_last_v2:
                self.log_file = os.path.join(self.training_opt['log_dir'], 'log_new_v2.txt')
            else:
                self.log_file = os.path.join(self.training_opt['log_dir'], 'log.txt')
                
            if os.path.isfile(self.log_file):
                os.remove(self.log_file)
            self.logger.log_cfg(self.config)
        else:
            if 'KNNClassifier' in self.config['networks']['classifier']['def_file']:
                #self.load_model()
                self.load_model(self.config['model_dir'])
                if not self.networks['classifier'].initialized:
                    cfeats = self.get_knncentroids('train_plain')
                    print('===> Saving features to %s' % 
                          os.path.join(self.training_opt['log_dir'], 'cfeats.pkl'))
                    with open(os.path.join(self.training_opt['log_dir'], 'cfeats.pkl'), 'wb') as f:
                        pickle.dump(cfeats, f)
                    self.networks['classifier'].update(cfeats)
            #self.log_file = None
            self.log_file = os.path.join(self.training_opt['log_dir'], 'log.txt')
            if os.path.isfile(self.log_file):
                os.remove(self.log_file)
            self.logger.log_cfg(self.config)
        
    def init_models(self, optimizer=True):
        networks_defs = self.config['networks']
        self.networks = {}
        self.model_optim_params_list = []

        print("Using", torch.cuda.device_count(), "GPUs.")
        
        for key, val in networks_defs.items():

            # Networks
            def_file = val['def_file']
            # model_args = list(val['params'].values())
            # model_args.append(self.test_mode)
            model_args = val['params']
            model_args.update({'test': self.test_mode})
            
            self.networks[key] = source_import(def_file).create_model(**model_args)
            if 'KNNClassifier' in type(self.networks[key]).__name__:
                # Put the KNN classifier on one single GPU
                self.networks[key] = self.networks[key].cuda()
            else:
                self.networks[key] = nn.DataParallel(self.networks[key]).cuda()

            if 'fix' in val and val['fix']:
                print('Freezing feature weights except for self attention weights (if exist).')
                for param_name, param in self.networks[key].named_parameters():
                    # Freeze all parameters except self attention parameters
                    if 'selfatt' not in param_name and 'fc' not in param_name:
                        param.requires_grad = False
                    # print('  | ', param_name, param.requires_grad)

            # Optimizer list
            optim_params = val['optim_params']
            self.model_optim_params_list.append({'params': self.networks[key].parameters(),
                                                'lr': optim_params['lr'],
                                                'momentum': optim_params['momentum'],
                                                'weight_decay': optim_params['weight_decay']})

    def init_criterions(self):
        criterion_defs = self.config['criterions']
        self.criterions = {}
        self.criterion_weights = {}

        for key, val in criterion_defs.items():
            def_file = val['def_file']
            loss_args = val['loss_params']
            
            if key == 'PerformanceLoss':
                self.drw = val['drw'] if 'drw' in val else False
                if 'LDAMLoss' in def_file:
                    loss_args['cls_num_list'] = self.data['train'].dataset.get_cls_num_list()
                    self.ldam = True
                else:
                    self.ldam = False
            
            self.criterions[key] = source_import(def_file).create_loss(**loss_args).cuda()
            self.criterion_weights[key] = val['weight']

            if val['optim_params']:
                print('Initializing criterion optimizer.')
                optim_params = val['optim_params']
                optim_params = [{'params': self.criterions[key].parameters(),
                                'lr': optim_params['lr'],
                                'momentum': optim_params['momentum'],
                                'weight_decay': optim_params['weight_decay']}]
                # Initialize criterion optimizer and scheduler
                self.criterion_optimizer, \
                self.criterion_optimizer_scheduler = self.init_optimizers(optim_params)
            else:
                self.criterion_optimizer = None

    def init_optimizers(self, optim_params):
        optimizer = optim.SGD(optim_params)
        if 'coslr' in self.config and self.config['coslr']:
            print("===> Using coslr eta_min={}".format(self.config['endlr']))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.training_opt['num_epochs'], eta_min=self.config['endlr'])
        elif 'warmup' in self.config and self.config['warmup']:
            print("===> Using warmup and multi-step learning rate.")
            scheduler = WarmupMultiStepLR(optimizer,
                                          milestones=self.scheduler_params['milestones'],
                                          gamma=self.scheduler_params['gamma'],
                                          warmup_epochs=self.scheduler_params['warmup_epochs'])
        elif self.scheduler_params is not None:
            scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=self.scheduler_params['step_size'],
                                                  gamma=self.scheduler_params['gamma'])
        else:
            scheduler = None

        return optimizer, scheduler

    def batch_forward (self, inputs, labels=None, centroids=False, feature_ext=False, phase='train'):
        '''
        This is a general single batch running function. 
        '''

        # Calculate Features
        self.features, self.feature_maps = self.networks['feat_model'](inputs)

        # If not just extracting features, calculate logits
        if not feature_ext:

            # During training, calculate centroids if needed to 
            if phase != 'test':
                if centroids and 'FeatureLoss' in self.criterions.keys():
                    self.centroids = self.criterions['FeatureLoss'].centroids.data
                    torch.cat([self.centroids] * self.num_gpus)
                else:
                    self.centroids = None

            if self.centroids is not None:
                centroids_ = torch.cat([self.centroids] * self.num_gpus)
            else:
                centroids_ = self.centroids
            
            # Calculate logits with classifier
            self.logits, self.direct_memory_feature = self.networks['classifier'](self.features, centroids_)
            
    def batch_backward(self):
        # Zero out optimizer gradients
        self.model_optimizer.zero_grad()
        if self.criterion_optimizer:
            self.criterion_optimizer.zero_grad()
        # Back-propagation from loss outputs
        self.loss.backward()
        # Step optimizers
        self.model_optimizer.step()
        if self.criterion_optimizer:
            self.criterion_optimizer.step()

    def batch_loss(self, labels, indexes):
        self.loss = 0

        # First, apply performance loss
        if 'PerformanceLoss' in self.criterions.keys():
            # self.loss_perf = self.criterions['PerformanceLoss'](self.logits, labels)
            self.loss_perf = sum([ soft_weight * self.criterions['PerformanceLoss'](self.logits, soft_target[indexes])
                                   for soft_weight, soft_target in zip(self.soft_weights, self.soft_targets) ])
            self.loss_perf *= self.criterion_weights['PerformanceLoss']
            self.loss += self.loss_perf

        # Apply loss on features if set up
        if 'FeatureLoss' in self.criterions.keys():
            self.loss_feat = self.criterions['FeatureLoss'](self.features, labels)
            self.loss_feat = self.loss_feat * self.criterion_weights['FeatureLoss']
            # Add feature loss to total loss
            self.loss += self.loss_feat
    
    def shuffle_batch(self, x, y, z=None):
        index = torch.randperm(x.size(0))
        x = x[index]
        y = y[index]
        if z == None:
            return x, y
        else:
            z = z[index]
            return x, y, z

    def train(self):
        # When training the network
        print_str = ['Phase: train']
        print_write(print_str, self.log_file)
        time.sleep(0.25)

        print_write(['Do shuffle??? --- ', self.do_shuffle], self.log_file)

        # Initialize best model
        best_model_weights = {}
        best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
        best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())
        ncm_best_model_weights = {}
        ncm_best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
        ncm_best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())
        best_acc = 0.0
        best_epoch = 0
        ncm_best_acc = 0.0
        ncm_best_epoch = 0
        # best_centroids = self.centroids
        
        end_epoch = self.training_opt['num_epochs']

        self.init_label_cleaning()

        # Loop over epochs
        for epoch in range(self.last_epoch + 1, end_epoch + 1):
            # Do label cleaning
            if self.do_cleaning:
                self.label_cleaning(epoch)
            
            for model in self.networks.values():
                model.train()

            torch.cuda.empty_cache()
            
            # Set model modes and set scheduler
            # In training, step optimizer scheduler and set model to train() 
            self.model_optimizer_scheduler.step()
            if self.criterion_optimizer:
                self.criterion_optimizer_scheduler.step()
            
            current_cls_num_list = np.array([(self.current_labels == i).sum().item() \
                                             for i in range(self.training_opt['num_classes'])])

            if self.ldam:
                self.criterions['PerformanceLoss'].update_cls_num_list(current_cls_num_list)

            # Use DRW rule
            if epoch == 161 and self.drw:
                print_str = ['Apply DRW', '\n']
                print_write(print_str, self.log_file)
                per_cls_weights = get_per_cls_weights(current_cls_num_list, beta=0.9999)
                self.criterions['PerformanceLoss'].weight = torch.FloatTensor(per_cls_weights).cuda()
                
            # Iterate over dataset
            total_preds = []
            total_labels = []

            for step, (inputs, labels, indexes) in enumerate(self.data['train']):
                # Break when step equal to epoch step
                if step == self.epoch_steps:
                    break
                # if self.do_shuffle:
                #     inputs, labels, indexes = self.shuffle_batch(inputs, labels, indexes)
                inputs, labels, indexes = inputs.cuda(), labels.cuda(), indexes.cuda()

                labels = self.current_labels[indexes]

                # If on training phase, enable gradients
                with torch.set_grad_enabled(True):
                        
                    # If training, forward with loss, and no top 5 accuracy calculation
                    self.batch_forward(inputs, labels, 
                                       centroids=self.memory['centroids'],
                                       phase='train')
                    self.batch_loss(labels, indexes)
                    self.batch_backward()

                    # Tracking predictions
                    _, preds = torch.max(self.logits, 1)
                    total_preds.append(torch2numpy(preds))
                    total_labels.append(torch2numpy(labels))

                    # Output minibatch training results
                    if step % self.training_opt['display_step'] == 0:

                        minibatch_loss_feat = self.loss_feat.item() \
                            if 'FeatureLoss' in self.criterions.keys() else None
                        minibatch_loss_perf = self.loss_perf.item() \
                            if 'PerformanceLoss' in self.criterions else None
                        minibatch_loss_total = self.loss.item()
                        minibatch_acc = mic_acc_cal(preds, labels)

                        print_str = ['Epoch: [%d/%d]' 
                                     % (epoch, self.training_opt['num_epochs']),
                                     'Step: %5d' 
                                     % (step),
                                     'Minibatch_loss_feature: %.4f' 
                                     % (minibatch_loss_feat) if minibatch_loss_feat else '',
                                     'Minibatch_loss_performance: %.4f'
                                     % (minibatch_loss_perf) if minibatch_loss_perf else '',
                                     'Minibatch_accuracy_micro: %.4f'
                                      % (minibatch_acc)]
                        print_write(print_str, self.log_file)

                        loss_info = {
                            'Epoch': epoch,
                            'Step': step,
                            'Total': minibatch_loss_total,
                            'CE': minibatch_loss_perf,
                            'feat': minibatch_loss_feat
                        }

                        self.logger.log_loss(loss_info)

                # Update priority weights if using PrioritizedSampler
                # if self.training_opt['sampler'] and \
                #    self.training_opt['sampler']['type'] == 'PrioritizedSampler':
                if hasattr(self.data['train'].sampler, 'update_weights'):
                    if hasattr(self.data['train'].sampler, 'ptype'):
                        ptype = self.data['train'].sampler.ptype 
                    else:
                        ptype = 'score'
                    ws = get_priority(ptype, self.logits.detach(), labels)
                    # ws = logits2score(self.logits.detach(), labels)
                    inlist = [indexes.cpu().numpy(), ws]
                    if self.training_opt['sampler']['type'] == 'ClassPrioritySampler':
                        inlist.append(labels.cpu().numpy())
                    self.data['train'].sampler.update_weights(*inlist)
                    # self.data['train'].sampler.update_weights(indexes.cpu().numpy(), ws)

            if hasattr(self.data['train'].sampler, 'get_weights'):
                self.logger.log_ws(epoch, self.data['train'].sampler.get_weights())
            if hasattr(self.data['train'].sampler, 'reset_weights'):
                self.data['train'].sampler.reset_weights(epoch)

            # After every epoch, validation
            rsls = {'epoch': epoch}
            rsls_train = self.eval_with_preds(total_preds, total_labels)
            rsls_eval = self.eval(phase='val')
            rsls.update(rsls_train)
            rsls.update(rsls_eval)

            self.eval_ncm(phase='val')

            # Reset class weights for sampling if pri_mode is valid
            if hasattr(self.data['train'].sampler, 'reset_priority'):
                ws = get_priority(self.data['train'].sampler.ptype,
                                  self.total_logits.detach(),
                                  self.total_labels)
                self.data['train'].sampler.reset_priority(ws, self.total_labels.cpu().numpy())

            # Log results
            self.logger.log_acc(rsls)

            # Under validation, the best model need to be updated
            if self.eval_acc_mic_top1 > best_acc:
                best_epoch = epoch
                best_acc = self.eval_acc_mic_top1
                best_centroids = self.centroids
                best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
                best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())
            
            if self.ncm_eval_acc_mic_top1 > ncm_best_acc:
                ncm_best_epoch = epoch
                ncm_best_acc = self.ncm_eval_acc_mic_top1
                ncm_best_centroids = self.centroids
                ncm_best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
                ncm_best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())

            print('===> Saving checkpoint')
            self.save_latest(epoch)

            if self.do_cleaning and epoch == 80:
                print('===> Saving as last model')
                self.save_last_model(epoch)
            if self.do_cleaning and epoch == 160:
                print('===> Saving as last_v2 model')
                self.save_last_model(epoch, 'last_v2')

        print()
        print('Training Complete.')

        print_str = ['Best validation accuracy is %.4f at epoch %d' % (best_acc, best_epoch)]
        print_write(print_str, self.log_file)
        print_str = ['NCM Best validation accuracy is %.4f at epoch %d' % (ncm_best_acc, ncm_best_epoch)]
        print_write(print_str, self.log_file)
        # Save the best model and best centroids if calculated
        self.save_model(epoch, best_epoch, best_model_weights, best_acc, centroids=best_centroids, tag='erm')
        self.save_model(epoch, ncm_best_epoch, ncm_best_model_weights, ncm_best_acc, centroids=ncm_best_centroids, tag='ncm')

        # Test on the test set
        self.reset_model(best_model_weights)
        self.eval('test' if 'test' in self.data else 'val')
        self.reset_model(ncm_best_model_weights)
        self.eval_ncm('test' if 'test' in self.data else 'val')
        print('Done')

        __f = open(os.path.join(self.training_opt['log_dir'], 'BestAcc_%.4f'%best_acc), 'wb')
        __f.close()
    
    def init_label_cleaning(self):
        self.noisy_labels = torch.LongTensor(self.data['train_plain'].dataset.targets).cuda()
        self.clean_labels = torch.LongTensor(self.data['train_plain'].dataset.clean_targets).cuda()

        # self.current_labels = torch.empty(0, dtype=torch.long).cuda()
        # for inputs, labels, paths in tqdm(self.data['train_plain']):
        #     labels = labels.cuda()
        #     self.current_labels = torch.cat((self.current_labels, labels))
        # assert (self.current_labels == self.noisy_labels).all()
        self.current_labels = self.noisy_labels

        self.cls_num_list = torch.LongTensor([(self.noisy_labels == i).sum().item() \
                                for i in range(self.training_opt['num_classes'])]).cuda()

        self.many_shot, self.few_shot = self.cls_num_list > 100, self.cls_num_list < 20
        self.medium_shot = ~(self.many_shot | self.few_shot)
        if self.config['training_opt']['dataset'] == 'ImbalanceCifar10':
            self.shots = torch.eye(self.training_opt['num_classes']).bool().cuda()
        else:
            self.shots = (self.many_shot, self.medium_shot, self.few_shot)

        self.soft_targets = [self.noisy_labels]
        self.soft_weights = [1]

    def label_cleaning(self, epoch):

        torch.cuda.empty_cache()
        # In validation or testing mode, set model to eval() and initialize running loss/correct
        for model in self.networks.values():
            model.eval()
        
        total_features = torch.empty((0, self.training_opt['feature_dim'])).cuda()
        total_logits = torch.empty((0, self.training_opt['num_classes'])).cuda()

        # Iterate over dataset
        for inputs, labels, paths in tqdm(self.data['train_plain']):
            inputs, labels = inputs.cuda(), labels.cuda()

            # If on training phase, enable gradients
            with torch.set_grad_enabled(False):

                # In validation or testing
                self.batch_forward(inputs, labels, 
                                   centroids=self.memory['centroids'],
                                   phase='train_plain')

                total_features = torch.cat((total_features, self.features))
                total_logits = torch.cat((total_logits, self.logits))

        cfeats = self.get_knncentroids(feats=total_features, labels=self.noisy_labels)
        self.ncm_classifier.update(cfeats)
        ncm_logits = self.ncm_classifier(total_features, None)[0]

        refined_ncm_logits = ncm_logits
        refine_times = 1
        for _ in range(refine_times):
            mask = self.get_gmm_mask(refined_ncm_logits)
            refined_cfeats = self.get_knncentroids(feats=total_features, labels=self.noisy_labels, mask=mask)
            self.ncm_classifier.update(refined_cfeats)
            refined_ncm_logits = self.ncm_classifier(total_features, None)[0]

        alpha = 0.6
        if epoch == 1:
            self.ensemble_erm_logits = total_logits
            self.ensemble_ncm_logits = refined_ncm_logits
        else:
            self.ensemble_erm_logits = alpha * self.ensemble_erm_logits + (1. - alpha) * total_logits
            self.ensemble_ncm_logits = alpha * self.ensemble_ncm_logits + (1. - alpha) * refined_ncm_logits
        self.ensemble_erm_logits = self.ensemble_erm_logits * (1. / (1. - alpha ** epoch))
        self.ensemble_ncm_logits = self.ensemble_ncm_logits * (1. / (1. - alpha ** epoch))

        # def plot_dist(ncm_logits_, tag=''):
        #     for i in range(self.training_opt['num_classes']):
        #         inner_logits_ = ncm_logits_[(self.noisy_labels == i) & (self.noisy_labels == self.clean_labels), i].cpu().numpy()
        #         outer_logits_ = ncm_logits_[(self.noisy_labels == i) & (self.noisy_labels != self.clean_labels), i].cpu().numpy()
        #         total_logits_ = ncm_logits_[self.noisy_labels == i, i].cpu().numpy()
        #         # plt.subplot(2, 5, i+1)
        #         plt.figure(figsize=(5, 3))
        #         sns.distplot(inner_logits_, kde=False, color='steelblue', label='clean'+'_'+str(i))
        #         sns.distplot(outer_logits_, kde=False, color='firebrick', label='noisy'+'_'+str(i))
        #         plt.legend(fontsize=15)
        #         plt.savefig(f'./assets/ncm_logits_e{epoch}_{i}_{tag}.png')
        # plot_dist(ncm_logits, 'raw')
        # plot_dist(refined_ncm_logits, 'refined')
        # plot_dist(self.ensemble_ncm_logits, 'ensemble')

        erm_outputs = self.ensemble_erm_logits.softmax(dim=1)
        ncm_outputs = self.ensemble_ncm_logits.softmax(dim=1)

        topk = 1
        print_str = [f'Top-{topk} Clean Precisions & Recalls:']
        def get_cls_clean_metrics(method, outputs, k):
            precisions, recalls = [], []
            probs, preds = outputs.topk(k=k)
            for shot in self.shots:
                p_idxs = shot[preds].sum(dim=1).bool()
                r_idxs = shot[self.clean_labels]
                p = (preds[p_idxs] == self.clean_labels[p_idxs].view(-1, 1)).sum(dim=1).bool().float().mean().item() if p_idxs.any() else 0.0
                r = (preds[r_idxs] == self.clean_labels[r_idxs].view(-1, 1)).sum(dim=1).bool().float().mean().item()
                precisions.append( round(p, 2) )
                recalls.append( round(r, 2) )
            print_str.extend(['\n', method, precisions, recalls])
        get_cls_clean_metrics('ERM: ', erm_outputs, topk)
        get_cls_clean_metrics('NCM: ', ncm_outputs, topk)
        print_str.append('\n')
        print_write(print_str, self.log_file)

        self.erm_probs, self.erm_preds = erm_outputs.max(dim=1)
        self.ncm_probs, self.ncm_preds = ncm_outputs.max(dim=1)
        
        self.mask = self.get_gmm_mask(refined_ncm_logits)

        shot_remains = [((self.mask & shot[self.noisy_labels]                          ).sum().item(), \
                         (self.mask & shot[self.noisy_labels] & shot[self.clean_labels]).sum().item(), \
                         (            shot[self.noisy_labels] & shot[self.clean_labels]).sum().item()) for shot in self.shots]
        
        print_str = ['Shot remains (all, clean, target):\n', shot_remains, '\n']
        print_write(print_str, self.log_file)

        surrogate_labels = torch.where(self.mask, self.noisy_labels, self.erm_preds)

        surrogate_noise_ratio = (surrogate_labels != self.clean_labels).float().mean().item()
        surrogate_shot_nums = [((shot[surrogate_labels] & (surrogate_labels != self.clean_labels)).sum().item(), \
                                 shot[surrogate_labels].sum().item() ) for shot in self.shots]
        print_str = ['Surrogate noise ratio:', surrogate_noise_ratio, '\n', surrogate_shot_nums, '\n']
        print_write(print_str, self.log_file)

        if self.drw and epoch > 160:
            if epoch == 161:
                self.current_labels = surrogate_labels

            self.soft_targets = [self.current_labels]
            self.soft_targets.extend( [torch.where(self.mask, self.noisy_labels, torch.ones_like(self.noisy_labels).long() * i)
                                       for i in range(self.training_opt['num_classes']) ])
            
            self.soft_weights = [0.5]
            self.soft_weights.extend([ 0.5 / self.training_opt['num_classes']
                                       for i in range(self.training_opt['num_classes']) ])

        elif epoch > 80:
            if self.ldam:
                self.current_labels = surrogate_labels

            self.soft_targets = [torch.where(self.mask, self.noisy_labels, self.erm_preds),
                                 torch.where(self.mask, self.noisy_labels, self.ncm_preds),
                                 torch.where(self.mask, self.noisy_labels, self.noisy_labels)]
            self.soft_targets.extend( [torch.where(self.mask, self.noisy_labels, torch.ones_like(self.noisy_labels).long() * i)
                                       for i in range(self.training_opt['num_classes']) ])
            
            self.soft_weights = [0.4, 0.2, 0.2]
            self.soft_weights.extend([ 0.2 / self.training_opt['num_classes']
                                       for i in range(self.training_opt['num_classes']) ])


    def get_gmm_mask(self, ncm_logits):
        mask = torch.zeros_like(self.noisy_labels).bool()

        for i in range(self.training_opt['num_classes']):
            this_cls_idxs = (self.noisy_labels == i)
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

    def eval_with_preds(self, preds, labels):
        # Count the number of examples
        n_total = sum([len(p) for p in preds])

        # Split the examples into normal and mixup
        normal_preds, normal_labels = [], []
        mixup_preds, mixup_labels1, mixup_labels2, mixup_ws = [], [], [], []
        for p, l in zip(preds, labels):
            if isinstance(l, tuple):
                mixup_preds.append(p)
                mixup_labels1.append(l[0])
                mixup_labels2.append(l[1])
                mixup_ws.append(l[2] * np.ones_like(l[0]))
            else:
                normal_preds.append(p)
                normal_labels.append(l)
        
        # Calculate normal prediction accuracy
        rsl = {'train_all':0., 'train_many':0., 'train_median':0., 'train_low': 0.}
        if len(normal_preds) > 0:
            normal_preds, normal_labels = list(map(np.concatenate, [normal_preds, normal_labels]))
            n_top1 = mic_acc_cal(normal_preds, normal_labels)
            n_top1_many, \
            n_top1_median, \
            n_top1_low, = shot_acc(normal_preds, normal_labels, self.data['train'])
            rsl['train_all'] += len(normal_preds) / n_total * n_top1
            rsl['train_many'] += len(normal_preds) / n_total * n_top1_many
            rsl['train_median'] += len(normal_preds) / n_total * n_top1_median
            rsl['train_low'] += len(normal_preds) / n_total * n_top1_low

        # Calculate mixup prediction accuracy
        if len(mixup_preds) > 0:
            mixup_preds, mixup_labels, mixup_ws = \
                list(map(np.concatenate, [mixup_preds*2, mixup_labels1+mixup_labels2, mixup_ws]))
            mixup_ws = np.concatenate([mixup_ws, 1-mixup_ws])
            n_top1 = weighted_mic_acc_cal(mixup_preds, mixup_labels, mixup_ws)
            n_top1_many, \
            n_top1_median, \
            n_top1_low, = weighted_shot_acc(mixup_preds, mixup_labels, mixup_ws, self.data['train'])
            rsl['train_all'] += len(mixup_preds) / 2 / n_total * n_top1
            rsl['train_many'] += len(mixup_preds) / 2 / n_total * n_top1_many
            rsl['train_median'] += len(mixup_preds) / 2 / n_total * n_top1_median
            rsl['train_low'] += len(mixup_preds) / 2 / n_total * n_top1_low

        # Top-1 accuracy and additional string
        print_str = ['\n Training acc Top1: %.4f \n' % (rsl['train_all']),
                     'Many_top1: %.4f' % (rsl['train_many']),
                     'Median_top1: %.4f' % (rsl['train_median']),
                     'Low_top1: %.4f' % (rsl['train_low']),
                     '\n']
        print_write(print_str, self.log_file)

        return rsl

    def eval(self, phase='val', openset=False, save_feat=False):

        print_str = ['Phase: %s' % (phase)]
        print_write(print_str, self.log_file)
        time.sleep(0.25)

        if openset:
            print('Under openset test mode. Open threshold is %.1f' 
                  % self.training_opt['open_threshold'])
 
        torch.cuda.empty_cache()

        # In validation or testing mode, set model to eval() and initialize running loss/correct
        for model in self.networks.values():
            model.eval()

        self.total_logits = torch.empty((0, self.training_opt['num_classes'])).cuda()
        self.total_labels = torch.empty(0, dtype=torch.long).cuda()
        self.total_paths = np.empty(0)

        get_feat_only = save_feat
        feats_all, labels_all, idxs_all, logits_all = [], [], [], []
        featmaps_all = []
        # Iterate over dataset
        for inputs, labels, paths in tqdm(self.data[phase]):
            inputs, labels = inputs.cuda(), labels.cuda()

            # If on training phase, enable gradients
            with torch.set_grad_enabled(False):

                # In validation or testing
                self.batch_forward(inputs, labels, 
                                   centroids=self.memory['centroids'],
                                   phase=phase)
                if not get_feat_only:
                    self.total_logits = torch.cat((self.total_logits, self.logits))
                    self.total_labels = torch.cat((self.total_labels, labels))
                    self.total_paths = np.concatenate((self.total_paths, paths))

                if get_feat_only:
                    logits_all.append(self.logits.cpu().numpy())
                    feats_all.append(self.features.cpu().numpy())
                    labels_all.append(labels.cpu().numpy())
                    idxs_all.append(paths.numpy())

        if get_feat_only:
            typ = 'feat'
            if phase == 'train_plain':
                name = 'train{}_all.pkl'.format(typ)
            elif phase == 'test':
                name = 'test{}_all.pkl'.format(typ)
            elif phase == 'val':
                name = 'val{}_all.pkl'.format(typ)

            fname = os.path.join(self.training_opt['log_dir'], name)
            print('===> Saving feats to ' + fname)
            with open(fname, 'wb') as f:
                pickle.dump({
                             'feats': np.concatenate(feats_all),
                             'labels': np.concatenate(labels_all),
                             'idxs': np.concatenate(idxs_all),
                            },
                            f, protocol=4) 
            return 
        probs, preds = F.softmax(self.total_logits.detach(), dim=1).max(dim=1)

        if openset:
            preds[probs < self.training_opt['open_threshold']] = -1
            self.openset_acc = mic_acc_cal(preds[self.total_labels == -1],
                                            self.total_labels[self.total_labels == -1])
            print('\n\nOpenset Accuracy: %.4f' % self.openset_acc)

        # Calculate the overall accuracy and F measurement
        self.eval_acc_mic_top1= mic_acc_cal(preds[self.total_labels != -1],
                                            self.total_labels[self.total_labels != -1])
        self.eval_f_measure = F_measure(preds, self.total_labels, openset=openset,
                                        theta=self.training_opt['open_threshold'])
        self.many_acc_top1, \
        self.median_acc_top1, \
        self.low_acc_top1, \
        self.cls_accs = shot_acc(preds[self.total_labels != -1],
                                 self.total_labels[self.total_labels != -1], 
                                 self.data['train'],
                                 acc_per_cls=True)

        self.cls_precisions, \
        self.cls_recalls = shot_precision_and_recall(preds[self.total_labels != -1],
                                                     self.total_labels[self.total_labels != -1])
        
        # Top-1 accuracy and additional string
        print_str = ['\n\n',
                     'Phase: %s' 
                     % (phase),
                     '\n\n',
                     'Evaluation_accuracy_micro_top1: %.4f' 
                     % (self.eval_acc_mic_top1),
                     '\n',
                     'Averaged F-measure: %.4f' 
                     % (self.eval_f_measure),
                     '\n',
                     'Many_shot_accuracy_top1: %.4f' 
                     % (self.many_acc_top1),
                     'Median_shot_accuracy_top1: %.4f' 
                     % (self.median_acc_top1),
                     'Low_shot_accuracy_top1: %.4f' 
                     % (self.low_acc_top1),
                     '\n']
        
        rsl = {phase + '_all': self.eval_acc_mic_top1,
               phase + '_many': self.many_acc_top1,
               phase + '_median': self.median_acc_top1,
               phase + '_low': self.low_acc_top1,
               phase + '_fscore': self.eval_f_measure}
        
        if 'KNNClassifier' in self.config['networks']['classifier']['def_file']:
            __f = open(os.path.join(self.training_opt['log_dir'], 'Acc_%.4f'%self.eval_acc_mic_top1), 'wb')
            __f.close()

            if self.config['training_opt']['dataset'] == 'ImbalanceCifar10':
                print_str.append('Class Accs: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]'
                                 % tuple(self.cls_accs))
                print_str.append('\n')
                print_str.append('Class Precisions: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]'
                                 % tuple(self.cls_precisions))
                print_str.append('\n')
                print_str.append('Class Recalls: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]'
                                 % tuple(self.cls_recalls))
                print_str.append('\n')
            print_write(print_str, self.log_file)

        if phase == 'val':
            if self.config['training_opt']['dataset'] == 'ImbalanceCifar10':
                print_str.append('Class Accs: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]'
                                 % tuple(self.cls_accs))
                print_str.append('\n')
                print_str.append('Class Precisions: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]'
                                 % tuple(self.cls_precisions))
                print_str.append('\n')
                print_str.append('Class Recalls: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]'
                                 % tuple(self.cls_recalls))
                print_str.append('\n')
            # elif self.config['training_opt']['dataset'] == 'ImbalanceCifar100':
            #     print_str.extend(['Class Accs:', list(self.cls_accs), '\n'])
            #     print_str.extend(['Class Precisions:', list(self.cls_precisions), '\n'])
            #     print_str.extend(['Class Recalls:', list(self.cls_recalls), '\n'])
            print_write(print_str, self.log_file)

        else:
            acc_str = ["{:.1f} \t {:.1f} \t {:.1f} \t {:.1f}".format(
                self.many_acc_top1 * 100,
                self.median_acc_top1 * 100,
                self.low_acc_top1 * 100,
                self.eval_acc_mic_top1 * 100)]
            if self.log_file is not None and os.path.exists(self.log_file):
                print_write(print_str, self.log_file)
                print_write(acc_str, self.log_file)
            else:
                print(*print_str)
                print(*acc_str)

        if phase == 'test':
            with open(os.path.join(self.training_opt['log_dir'], 'cls_accs.pkl'), 'wb') as f:
                pickle.dump(self.cls_accs, f)
            with open(os.path.join(self.training_opt['log_dir'], 'cls_precisions.pkl'), 'wb') as f:
                pickle.dump(self.cls_precisions, f)
            with open(os.path.join(self.training_opt['log_dir'], 'cls_recalls.pkl'), 'wb') as f:
                pickle.dump(self.cls_recalls, f)
        return rsl

    def eval_ncm(self, phase='val', openset=False):

        print_str = ['Phase: %s' % (phase)]
        print_write(print_str, self.log_file)
        time.sleep(0.25)
 
        torch.cuda.empty_cache()

        # In validation or testing mode, set model to eval() and initialize running loss/correct
        for model in self.networks.values():
            model.eval()

        total_features = torch.empty((0, self.training_opt['feature_dim'])).cuda()
        total_labels = torch.empty(0, dtype=torch.long).cuda()

        featmaps_all = []
        # Iterate over dataset
        for inputs, labels, paths in tqdm(self.data[phase]):
            inputs, labels = inputs.cuda(), labels.cuda()

            # If on training phase, enable gradients
            with torch.set_grad_enabled(False):

                # In validation or testing
                self.batch_forward(inputs, labels, 
                                   centroids=self.memory['centroids'],
                                   phase=phase)
                total_features = torch.cat((total_features, self.features))
                total_labels = torch.cat((total_labels, labels))

        if not self.do_cleaning:
            cfeats = self.get_knncentroids('train_plain')
            self.ncm_classifier.update(cfeats)
        ncm_logits = self.ncm_classifier(total_features, None)[0]
        ncm_probs, ncm_preds = F.softmax(ncm_logits.detach(), dim=1).max(dim=1)

        # Calculate the overall accuracy and F measurement
        self.ncm_eval_acc_mic_top1= mic_acc_cal(ncm_preds[total_labels != -1],
                                            total_labels[total_labels != -1])
        self.ncm_eval_f_measure = F_measure(ncm_preds, total_labels, openset=openset,
                                        theta=self.training_opt['open_threshold'])
        self.ncm_many_acc_top1, \
        self.ncm_median_acc_top1, \
        self.ncm_low_acc_top1, \
        self.ncm_cls_accs = shot_acc(ncm_preds[total_labels != -1],
                            total_labels[total_labels != -1], 
                            self.data['train'],
                            acc_per_cls=True)

        self.ncm_cls_precisions, \
        self.ncm_cls_recalls = shot_precision_and_recall(ncm_preds[total_labels != -1],
                                                total_labels[total_labels != -1])
        
        # Top-1 accuracy and additional string
        print_str = ['\n\n',
                     'Phase: %s' 
                     % (phase),
                     '\n\n',
                     'Evaluation_accuracy_micro_top1: %.4f' 
                     % (self.ncm_eval_acc_mic_top1),
                     '\n',
                     'Averaged F-measure: %.4f' 
                     % (self.ncm_eval_f_measure),
                     '\n',
                     'Many_shot_accuracy_top1: %.4f' 
                     % (self.ncm_many_acc_top1),
                     'Median_shot_accuracy_top1: %.4f' 
                     % (self.ncm_median_acc_top1),
                     'Low_shot_accuracy_top1: %.4f' 
                     % (self.ncm_low_acc_top1),
                     '\n']

        if phase == 'val':
            if self.config['training_opt']['dataset'] == 'ImbalanceCifar10':
                print_str.append('Class Accs: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]'
                                 % tuple(self.ncm_cls_accs))
                print_str.append('\n')
                print_str.append('Class Precisions: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]'
                                 % tuple(self.ncm_cls_precisions))
                print_str.append('\n')
                print_str.append('Class Recalls: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]'
                                 % tuple(self.ncm_cls_recalls))
                print_str.append('\n')
            # elif self.config['training_opt']['dataset'] == 'ImbalanceCifar100':
            #     print_str.extend(['Class Accs:', list(self.ncm_cls_accs), '\n'])
            #     print_str.extend(['Class Precisions:', list(self.ncm_cls_precisions), '\n'])
            #     print_str.extend(['Class Recalls:', list(self.ncm_cls_recalls), '\n'])
            print_write(print_str, self.log_file)
        else:
            acc_str = ["{:.1f} \t {:.1f} \t {:.1f} \t {:.1f}".format(
                self.ncm_many_acc_top1 * 100,
                self.ncm_median_acc_top1 * 100,
                self.ncm_low_acc_top1 * 100,
                self.ncm_eval_acc_mic_top1 * 100)]
            if self.log_file is not None and os.path.exists(self.log_file):
                print_write(print_str, self.log_file)
                print_write(acc_str, self.log_file)
            else:
                print(*print_str)
                print(*acc_str)

        if phase == 'test':
            with open(os.path.join(self.training_opt['log_dir'], 'cls_accs.pkl'), 'wb') as f:
                pickle.dump(self.ncm_cls_accs, f)
            with open(os.path.join(self.training_opt['log_dir'], 'cls_precisions.pkl'), 'wb') as f:
                pickle.dump(self.ncm_cls_precisions, f)
            with open(os.path.join(self.training_opt['log_dir'], 'cls_recalls.pkl'), 'wb') as f:
                pickle.dump(self.ncm_cls_recalls, f)


    def centroids_cal(self, data, save_all=False):

        centroids = torch.zeros(self.training_opt['num_classes'],
                                   self.training_opt['feature_dim']).cuda()

        print('Calculating centroids.')

        torch.cuda.empty_cache()
        for model in self.networks.values():
            model.eval()

        feats_all, labels_all, idxs_all = [], [], []

        # Calculate initial centroids only on training data.
        with torch.set_grad_enabled(False):
            for inputs, labels, idxs in tqdm(data):
                inputs, labels = inputs.cuda(), labels.cuda()

                # Calculate Features of each training data
                self.batch_forward(inputs, feature_ext=True)
                # Add all calculated features to center tensor
                for i in range(len(labels)):
                    label = labels[i]
                    centroids[label] += self.features[i]
                # Save features if requried
                if save_all:
                    feats_all.append(self.features.cpu().numpy())
                    labels_all.append(labels.cpu().numpy())
                    idxs_all.append(idxs.numpy())
        
        if save_all:
            fname = os.path.join(self.training_opt['log_dir'], 'feats_all.pkl')
            with open(fname, 'wb') as f:
                pickle.dump({'feats': np.concatenate(feats_all),
                             'labels': np.concatenate(labels_all),
                             'idxs': np.concatenate(idxs_all)},
                            f)
        # Average summed features with class count
        centroids /= torch.tensor(class_count(data)).float().unsqueeze(1).cuda()

        return centroids

    def get_knncentroids(self, datakey=None, feats=None, labels=None, mask=None):
        
        if feats is not None and labels is not None:
            feats = feats.cpu().numpy()
            labels = labels.cpu().numpy()
        else:
            # datakey = 'train_plain'
            assert datakey in self.data
            print('===> Calculating KNN centroids.')

            torch.cuda.empty_cache()
            for model in self.networks.values():
                model.eval()

            feats_all, labels_all = [], []

            # Calculate initial centroids only on training data.
            with torch.set_grad_enabled(False):
                for inputs, labels, idxs in tqdm(self.data[datakey]):
                    inputs, labels = inputs.cuda(), labels.cuda()

                    # Calculate Features of each training data
                    self.batch_forward(inputs, feature_ext=True)

                    feats_all.append(self.features.cpu().numpy())
                    labels_all.append(labels.cpu().numpy())
            
            feats = np.concatenate(feats_all)
            labels = np.concatenate(labels_all)

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
    
    def reset_model(self, model_state):
        for key, model in self.networks.items():
            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            model.load_state_dict(weights)

    def load_model(self, model_dir=None):
        model_dir = self.training_opt['log_dir'] if model_dir is None else model_dir
        if not model_dir.endswith('.pth'):
            model_dir = os.path.join(model_dir, 'final_model_checkpoint.pth')
        
        print('Validation on the best model.')
        print('Loading model from %s' % (model_dir))
        
        checkpoint = torch.load(model_dir)          
        model_state = checkpoint['state_dict_best']
        
        self.centroids = checkpoint['centroids'] if 'centroids' in checkpoint else None
        
        for key, model in self.networks.items():
            # if not self.test_mode and key == 'classifier':
            if not self.test_mode and \
                'DotProductClassifier' in self.config['networks'][key]['def_file']:
                # Skip classifier initialization 
                print('Skiping classifier initialization')
                continue
            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            x = model.state_dict()
            x.update(weights)
            model.load_state_dict(x)

    def save_latest(self, epoch):
        model_weights = {}
        model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
        model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())

        model_states = {
            'epoch': epoch,
            'state_dict': model_weights
        }

        model_dir = os.path.join(self.training_opt['log_dir'], 
                                 'latest_model_checkpoint.pth')
        torch.save(model_states, model_dir)

    def save_model(self, epoch, best_epoch, best_model_weights, best_acc, centroids=None, tag='erm'):
        
        model_states = {'epoch': epoch,
                'best_epoch': best_epoch,
                'state_dict_best': best_model_weights,
                'best_acc': best_acc,
                'centroids': centroids}
        
        if tag == 'erm':
            model_dir = os.path.join(self.training_opt['log_dir'], 
                                    'final_model_checkpoint.pth')
        elif tag == 'ncm':
            model_dir = os.path.join(self.training_opt['log_dir'], 
                                    'ncm_final_model_checkpoint.pth')

        torch.save(model_states, model_dir)

    def save_last_model(self, epoch, name='last'):
        model_weights = {}
        model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
        model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())

        model_states = {
            'epoch': epoch,
            'state_dict': model_weights,
            'optimizer': self.model_optimizer.state_dict(),
            'scheduler': self.model_optimizer_scheduler.state_dict(),
            'erm_logits': self.ensemble_erm_logits,
            'ncm_logits': self.ensemble_ncm_logits,
        }

        model_dir = os.path.join(self.training_opt['log_dir'], 
                                f'{name}_model_checkpoint.pth')
        torch.save(model_states, model_dir)
    
    def load_last_model(self, name='last'):
        model_dir = os.path.join(self.training_opt['log_dir'], 
                                 f'{name}_model_checkpoint.pth')

        print('Loading last trained model from %s' % (model_dir))

        checkpoint = torch.load(model_dir)
        self.last_epoch = checkpoint['epoch']

        model_state = checkpoint['state_dict']
        for key, model in self.networks.items():
            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            x = model.state_dict()
            x.update(weights)
            model.load_state_dict(x)

        self.model_optimizer.load_state_dict(checkpoint['optimizer'])
        self.model_optimizer_scheduler.load_state_dict(checkpoint['scheduler'])

        self.ensemble_erm_logits = checkpoint['erm_logits'] 
        self.ensemble_ncm_logits = checkpoint['ncm_logits']
            
    def output_logits(self, openset=False):
        filename = os.path.join(self.training_opt['log_dir'], 
                                'logits_%s'%('open' if openset else 'close'))
        print("Saving total logits to: %s.npz" % filename)
        np.savez(filename, 
                 logits=self.total_logits.detach().cpu().numpy(), 
                 labels=self.total_labels.detach().cpu().numpy(),
                 paths=self.total_paths)
