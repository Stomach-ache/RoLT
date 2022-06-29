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
import argparse
import pprint
import random
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from data import dataloader
from data.imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from run_networks import model
import warnings
import yaml
from utils import source_import, get_value

data_root = {'ImageNet': '/datasets01_101/imagenet_full_size/061417',
             'Places': '/datasets01_101/Places365/041019',
             'iNaturalist18': '/checkpoint/bykang/iNaturalist18',
             'Cifar': './data/Cifar'}

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None, type=str)
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--test_open', default=False, action='store_true')
parser.add_argument('--output_logits', default=False)
parser.add_argument('--model_dir', type=str, default=None)
parser.add_argument('--save_feat', type=str, default='')

# Imbalance Cifar parameters
parser.add_argument('--imb_type', default='exp', type=str)
parser.add_argument('--imb_factor', default=0.01, type=float)
parser.add_argument('--noise_mode', default='imb', type=str)
parser.add_argument('--noise_ratio', default=0.0, type=float)
parser.add_argument('--feat_noise_ratio', default=0.0, type=float)

# KNN testing parameters 
parser.add_argument('--knn', default=False, action='store_true')
parser.add_argument('--feat_type', type=str, default='cl2n')
parser.add_argument('--dist_type', type=str, default='l2')

# Learnable tau
parser.add_argument('--val_as_train', default=False, action='store_true')

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cleaning', default=False, action='store_true')
parser.add_argument('--load_last', default=False, action='store_true')
parser.add_argument('--load_last_v2', default=False, action='store_true')
parser.add_argument('--tag', type=str, default=None)

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

set_seed()

def update(config, args):
    # Change parameters
    config['model_dir'] = get_value(config['model_dir'], args.model_dir)
    config['training_opt']['batch_size'] = \
        get_value(config['training_opt']['batch_size'], args.batch_size)

    # Testing with KNN
    if args.knn and args.test:
        training_opt = config['training_opt']
        classifier_param = {
            'feat_dim': training_opt['feature_dim'],
            'num_classes': training_opt['num_classes'], 
            'feat_type': args.feat_type,
            'dist_type': args.dist_type,
            'log_dir': training_opt['log_dir']}
        classifier = {
            'def_file': './models/KNNClassifier.py',
            'params': classifier_param,
            'optim_params': config['networks']['classifier']['optim_params']}
        config['networks']['classifier'] = classifier

    config['cleaning'] = args.cleaning
    config['load_last'] = args.load_last
    config['load_last_v2'] = args.load_last_v2
    config['tag'] = args.tag

    return config

# ============================================================================
# LOAD CONFIGURATIONS
with open(args.cfg) as f:
    config = yaml.load(f)
config = update(config, args)

# log_suffix = '_' + args.imb_type + '_' + str(args.imb_factor) + '_' + args.noise_mode + '_' + str(args.noise_ratio)
# if config['model_dir'] is not None and args.model_dir is None:
#     model_suffix = '_' + args.imb_type + '_' + str(args.imb_factor) + '_' + args.noise_mode + '_' + str(args.feat_noise_ratio)
#     config['model_dir'] += model_suffix
#     log_suffix += '_' + '(' + str(args.feat_noise_ratio) + ')'
# if config['cleaning']:
#     log_suffix += '_cleaning'
# config['training_opt']['log_dir'] += log_suffix

log_suffix = f'_{args.imb_type}_{args.imb_factor}_{args.noise_mode}_{args.noise_ratio}'
if config['model_dir'] is not None and args.model_dir is None:
    model_suffix = f'_{args.imb_type}_{args.imb_factor}_{args.noise_mode}_{args.feat_noise_ratio}'
    if args.cleaning:
        model_suffix += '_cleaning'
    config['model_dir'] += model_suffix
    log_suffix += f'_({args.feat_noise_ratio})'
if args.cleaning:
    log_suffix += '_cleaning'
config['training_opt']['log_dir'] += log_suffix

test_mode = args.test
test_open = args.test_open
if test_open:
    test_mode = True
output_logits = args.output_logits
training_opt = config['training_opt']
relatin_opt = config['memory']
dataset = training_opt['dataset']

if not os.path.isdir(training_opt['log_dir']):
    os.makedirs(training_opt['log_dir'])

if 'Cifar' not in training_opt['dataset']:
    print('Loading dataset from: %s' % data_root[dataset.rstrip('_LT')])
else:
    print('Loading dataset from: %s' % data_root['Cifar'])
pprint.pprint(config)

def split2phase(split):
    if split == 'train' and args.val_as_train:
        return 'train_val'
    else:
        return split

def get_sampler_dic(split):
    sampler_defs = training_opt['sampler']
    if sampler_defs:
        if sampler_defs['type'] == 'ClassAwareSampler':
            sampler_dic = {
                'sampler': source_import(sampler_defs['def_file']).get_sampler(),
                'params': {'num_samples_cls': sampler_defs['num_samples_cls']}
            }
        elif sampler_defs['type'] in ['MixedPrioritizedSampler',
                                      'ClassPrioritySampler']:
            sampler_dic = {
                'sampler': source_import(sampler_defs['def_file']).get_sampler(),
                'params': {k: v for k, v in sampler_defs.items() \
                           if k not in ['type', 'def_file']}
            }
    else:
        sampler_dic = None
    return sampler_dic

def get_imbalance_cifar(dataset, args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if dataset.endswith('10'):
        train_dataset = IMBALANCECIFAR10(
            root='./data/Cifar',
            imb_type=args.imb_type, imb_factor=args.imb_factor,
            noise_mode=args.noise_mode, noise_ratio=args.noise_ratio,
            train=True, download=True, transform=transform_train
        )
        valid_dataset = IMBALANCECIFAR10(
            root='./data/Cifar',
            train=False, download=True, transform=transform_val
        )
    elif dataset.endswith('100'):
        train_dataset = IMBALANCECIFAR100(
            root='./data/Cifar',
            imb_type=args.imb_type, imb_factor=args.imb_factor,
            noise_mode=args.noise_mode, noise_ratio=args.noise_ratio,
            train=True, download=True, transform=transform_train
        )
        valid_dataset = IMBALANCECIFAR100(
            root='./data/Cifar',
            train=False, download=True, transform=transform_val
        )

    shuffle = config['shuffle'] if 'shuffle' in config else False
    config['shuffle'] = False
    trainLoader = DataLoader(train_dataset, batch_size=training_opt['batch_size'], shuffle=shuffle, num_workers=training_opt['num_workers'], pin_memory=True)
    plainLoader = DataLoader(train_dataset, batch_size=training_opt['batch_size'], shuffle=False, num_workers=training_opt['num_workers'], pin_memory=True)
    validLoader = DataLoader(valid_dataset, batch_size=training_opt['batch_size'], shuffle=False, num_workers=training_opt['num_workers'], pin_memory=True)

    return {'train': trainLoader, 'train_plain': plainLoader, 'val': validLoader}

if not test_mode:

    splits = ['train', 'train_plain', 'val']
    if dataset not in ['iNaturalist18', 'ImageNet']:
        splits.append('test')
    
    if 'Cifar' not in dataset:
        data = {x: dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')],
                                        dataset=dataset, phase=split2phase(x), 
                                        batch_size=training_opt['batch_size'],
                                        sampler_dic=get_sampler_dic(x),
                                        num_workers=training_opt['num_workers'])
                for x in splits}
    else:
        data = get_imbalance_cifar(dataset, args)

    training_model = model(config, data, test=False)

    training_model.train()

else:

    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data",
                            UserWarning)

    print('Under testing phase, we load training data simply to calculate \
           training data number for each class.')

    if 'iNaturalist' in training_opt['dataset']:
        splits = ['train', 'val']
        test_split = 'val'
    else:
        splits = ['train', 'val', 'test']
        test_split = 'test'
    if 'ImageNet' == training_opt['dataset']:
        splits = ['train', 'val']
        test_split = 'val'
    if args.knn or True:
        splits.append('train_plain')

    if 'Cifar' not in dataset:
        data = {x: dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')],
                                        dataset=dataset, phase=x,
                                        batch_size=training_opt['batch_size'],
                                        sampler_dic=None, 
                                        test_open=test_open,
                                        num_workers=training_opt['num_workers'],
                                        shuffle=False)
                for x in splits}
    else:
        data = get_imbalance_cifar(dataset, args)

    training_model = model(config, data, test=True)
    # training_model.load_model()
    #training_model.load_model(args.model_dir)
    training_model.load_model(config['model_dir'])
    if args.save_feat in ['train_plain', 'val', 'test']:
        saveit = True
        test_split = args.save_feat
    else:
        saveit = False
    
    training_model.eval(phase=test_split, openset=test_open, save_feat=saveit)
    
    if output_logits:
        training_model.output_logits(openset=test_open)
        
print('ALL COMPLETED.')
