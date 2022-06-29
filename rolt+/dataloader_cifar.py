from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class cifar_dataset(Dataset):
    def __init__(self, dataset, imb_type, imb_factor, noise_mode, noise_ratio, root_dir, transform, mode, noise_file='', pred=[], probability=[], log='', seed=0):

        np.random.seed(seed)
        random.seed(seed)

        self.noise_ratio = noise_ratio # noise ratio
        self.transform = transform
        self.mode = mode
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise

        if self.mode=='test':
            if dataset=='cifar10':
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['labels']
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['fine_labels']
        else:
            train_data=[]
            train_label=[]
            if dataset=='cifar10':
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label+data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset=='cifar100':
                train_dic = unpickle('%s/train'%root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))

            self.data = train_data
            self.targets = train_label
            self.cls_num = 10 if dataset == 'cifar10' else 100
            noise_ratio = self.noise_ratio

            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            imb_file = os.path.join(root_dir, 'cifar' + str(self.cls_num) + '_' + imb_type + '_' + str(imb_factor) + '_' + str(seed))
            self.gen_imbalanced_data(img_num_list, imb_file)

            noise_file = os.path.join(root_dir, 'cifar' + str(self.cls_num) + '_' + imb_type + '_' + str(imb_factor) + '_' + noise_mode + '_' + str(noise_ratio) + '_' + str(seed))
            self.get_noisy_data(self.cls_num, noise_file, noise_mode, noise_ratio)

            train_data = self.data
            train_label = self.clean_targets
            noise_label = self.targets

            # if os.path.exists(noise_file):
            #     noise_label = json.load(open(noise_file,"r"))
            # else:    #inject noise
            #     noise_label = []
            #     idx = list(range(50000))
            #     random.shuffle(idx)
            #     num_noise = int(self.noise_ratio*50000)
            #     noise_idx = idx[:num_noise]
            #     for i in range(50000):
            #         if i in noise_idx:
            #             if noise_mode=='sym':
            #                 if dataset=='cifar10':
            #                     noiselabel = random.randint(0,9)
            #                 elif dataset=='cifar100':
            #                     noiselabel = random.randint(0,99)
            #                 noise_label.append(noiselabel)
            #             elif noise_mode=='asym':
            #                 noiselabel = self.transition[train_label[i]]
            #                 noise_label.append(noiselabel)
            #         else:
            #             noise_label.append(train_label[i])
            #     print("save noisy labels to %s ..."%noise_file)
            #     json.dump(noise_label,open(noise_file,"w"))

            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
            else:
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]

                    clean = (np.array(noise_label)==np.array(train_label))
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(probability,clean)
                    auc,_,_ = auc_meter.value()

                    cls_id_list, cls_num_list = np.unique(noise_label, return_counts=True)
                    if dataset == 'cifar100':
                        many_shot = cls_num_list > 100
                        few_shot = cls_num_list < 20
                        medium_shot = ~(many_shot | few_shot)
                    elif dataset == 'cifar10':
                        many_shot = cls_id_list < 2
                        few_shot = cls_id_list >= 7
                        medium_shot = ~(many_shot | few_shot)

                    log.write('Numer of labeled samples:%d   AUC:%.3f   '%(pred.sum(),auc))
                    log.write('P&R:')
                    for shot in (many_shot, medium_shot, few_shot):
                        p_idxs = shot[noise_label] & pred
                        r_idxs = shot[noise_label] & clean
                        p = (pred[p_idxs] == clean[p_idxs]).mean() if p_idxs.any() else 0.0
                        r = (pred[r_idxs] == clean[r_idxs]).mean()
                        log.write('(%.4f %.4f) '%(p, r))
                    log.write('\n')

                    log.flush()

                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]

                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]
                self.pred_idx = pred_idx
                print("%s data has a size of %d"%(self.mode,len(self.noise_label)))

    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2, target, prob, self.pred_idx[index]
        elif self.mode=='unlabeled':
            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2, self.pred_idx[index]
        elif self.mode=='all':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target, index
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)


    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls, imb_file):
        targets_np = np.array(self.targets, dtype=np.int64)

        if os.path.exists(imb_file):
            imb_idx = json.load(open(imb_file,"r"))
            self.num_per_cls_dict = dict()
            for i in imb_idx:
                if self.targets[i] in self.num_per_cls_dict:
                    self.num_per_cls_dict[self.targets[i]] += 1
                else:
                    self.num_per_cls_dict[self.targets[i]] = 1
        else:
            imb_idx = []
            #new_data = []
            #new_targets = []
            classes = np.unique(targets_np)
            # np.random.shuffle(classes)
            self.num_per_cls_dict = dict()
            for the_class, the_img_num in zip(classes, img_num_per_cls):
                self.num_per_cls_dict[the_class] = the_img_num
                idx = np.where(targets_np == the_class)[0]
                np.random.shuffle(idx)
                selec_idx = idx[:the_img_num]
                imb_idx.extend(selec_idx)
                #new_data.append(self.data[selec_idx, ...])
                #new_targets.extend([the_class, ] * the_img_num)
            #new_data = np.vstack(new_data)

            imb_idx = np.array(imb_idx, dtype=np.int64).tolist()
            print("save imbalanced data index to %s ..." % imb_file)
            json.dump(imb_idx, open(imb_file,"w"))

        self.data = self.data[imb_idx]
        self.targets = targets_np[imb_idx]

    def get_noisy_data(self, cls_num, noise_file, noise_mode, noise_ratio):
        train_label = self.targets

        if os.path.exists(noise_file):
            noise_label = json.load(open(noise_file,"r"))
        else:    #inject noise
            #label_dict = {}
            #label_dict['clean_labels'] = train_label
            noise_label = []
            num_train = len(self.targets)
            idx = list(range(num_train))
            random.shuffle(idx)
            cls_num_list = self.get_cls_num_list()

            if noise_mode == 'sym':
                num_noise = int(noise_ratio * num_train)
                noise_idx = idx[:num_noise]

                for i in range(num_train):
                    if i in noise_idx:
                        newlabel = (random.randint(1, cls_num - 1) + train_label[i]) % cls_num
                        assert newlabel != train_label[i]
                        noise_label.append(newlabel)
                    else:
                        noise_label.append(train_label[i])

            elif noise_mode == 'imb':
                num_noise = int(noise_ratio * num_train)
                noise_idx = idx[:num_noise]

                p = np.array([cls_num_list for _ in range(cls_num)])
                for i in range(cls_num):
                    p[i][i] = 0
                p = p / p.sum(axis=1, keepdims=True)
                for i in range(num_train):
                    if i in noise_idx:
                        newlabel = np.random.choice(cls_num, p=p[train_label[i]])
                        assert newlabel != train_label[i]
                        noise_label.append(newlabel)
                    else:
                        noise_label.append(train_label[i])

            noise_label = np.array(noise_label, dtype=np.int8).tolist()
            #label_dict['noisy_labels'] = noise_label
            print("save noisy labels to %s ..." % noise_file)
            json.dump(noise_label, open(noise_file,"w"))

        self.clean_targets = self.targets[:]
        self.targets = noise_label

        for c1, c0 in zip(self.targets, self.clean_targets):
            if c1 != c0:
                self.num_per_cls_dict[c1] += 1
                self.num_per_cls_dict[c0] -= 1

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class cifar_dataloader():
    def __init__(self, dataset, imb_type, imb_factor, noise_mode, noise_ratio, batch_size, num_workers, root_dir, log, noise_file='', seed=0):
        self.dataset = dataset
        self.imb_type = imb_type
        self.imb_factor = imb_factor
        self.noise_mode = noise_mode
        self.noise_ratio = noise_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.seed = seed
        self.noise_file = noise_file
        if self.dataset=='cifar10':
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])
        elif self.dataset=='cifar100':
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])
    def run(self,mode,pred=[],prob=[]):
        if mode=='warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, imb_type=self.imb_type, imb_factor=self.imb_factor, noise_mode=self.noise_mode, noise_ratio=self.noise_ratio, root_dir=self.root_dir, transform=self.transform_train, mode="all",noise_file=self.noise_file, seed=self.seed)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)
            return trainloader

        elif mode=='train':
            labeled_dataset = cifar_dataset(dataset=self.dataset, imb_type=self.imb_type, imb_factor=self.imb_factor, noise_mode=self.noise_mode, noise_ratio=self.noise_ratio, root_dir=self.root_dir, transform=self.transform_train, mode="labeled", noise_file=self.noise_file, pred=pred, probability=prob,log=self.log, seed=self.seed)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)

            unlabeled_dataset = cifar_dataset(dataset=self.dataset, imb_type=self.imb_type, imb_factor=self.imb_factor, noise_mode=self.noise_mode, noise_ratio=self.noise_ratio, root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled", noise_file=self.noise_file, pred=pred, seed=self.seed)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            return labeled_trainloader, unlabeled_trainloader

        elif mode=='test':
            test_dataset = cifar_dataset(dataset=self.dataset, imb_type=self.imb_type, imb_factor=self.imb_factor, noise_mode=self.noise_mode, noise_ratio=self.noise_ratio, root_dir=self.root_dir, transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader

        elif mode=='eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, imb_type=self.imb_type, imb_factor=self.imb_factor, noise_mode=self.noise_mode, noise_ratio=self.noise_ratio, root_dir=self.root_dir, transform=self.transform_test, mode='all', noise_file=self.noise_file, seed=self.seed)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader
