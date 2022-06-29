from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import torch
import os
import json

class imagenet_dataset(Dataset):
    def __init__(self, root_dir, transform, num_class):
        self.root = root_dir+'imagenet/val/'
        self.transform = transform
        self.val_data = []
        for c in range(num_class):
            imgs = os.listdir(self.root+str(c))
            for img in imgs:
                self.val_data.append([c,os.path.join(self.root,str(c),img)])

    def __getitem__(self, index):
        data = self.val_data[index]
        target = data[0]
        image = Image.open(data[1]).convert('RGB')
        img = self.transform(image)
        return img, target

    def __len__(self):
        return len(self.val_data)

class webvision_dataset(Dataset):
    def __init__(self, root_dir, transform, mode, num_class, pred=[], probability=[], log='', refine_labels=None, imb_type='exp', imb_factor=1):
        self.root = root_dir
        self.transform = transform
        self.mode = mode

        if self.mode=='test':
            with open(self.root+'info/val_filelist.txt') as f:
                lines=f.readlines()
            self.val_imgs = []
            self.val_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_class:
                    self.val_imgs.append(img)
                    self.val_labels[img]=target
        else:
            with open(self.root+'info/train_filelist_google.txt') as f:
                lines=f.readlines()
            train_imgs = []
            self.train_labels = {}
            i = 0
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_class:
                    if refine_labels is not None:
                        target = refine_labels[i]
                        i += 1
                    train_imgs.append(img)
                    self.train_labels[img]=target

            self.cls_num = num_class

            self.train_data = np.array(train_imgs)
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            self.img_num_list = img_num_list
            #print (img_num_list)
            print (max(img_num_list), min(img_num_list), max(img_num_list) / min(img_num_list))
            print (sum(img_num_list))

            if imb_factor < 0.5:
                imb_file = os.path.join('.', 'webvision_' + str(imb_factor))
                self.gen_imbalanced_data(img_num_list, imb_file)
                train_imgs = self.new_train_data

            if self.mode == 'all':
                self.train_imgs = train_imgs
            else:
                if self.mode == "labeled":
                    #print (len(pred))
                    #print (len(train_imgs))
                    pred_idx = pred.nonzero()[0]
                    self.train_imgs = [train_imgs[i] for i in pred_idx]
                    self.probability = [probability[i] for i in pred_idx]
                    print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))
                    log.write('Numer of labeled samples:%d \n'%(pred.sum()))
                    log.flush()
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]
                    self.train_imgs = [train_imgs[i] for i in pred_idx]
                    print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))

    def __getitem__(self, index):
        if self.mode=='labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            prob = self.probability[index]
            image = Image.open(self.root+img_path).convert('RGB')
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2, target, prob
        elif self.mode=='unlabeled':
            img_path = self.train_imgs[index]
            image = Image.open(self.root+img_path).convert('RGB')
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2
        elif self.mode=='all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(self.root+img_path).convert('RGB')
            img = self.transform(image)
            return img, target, index
        elif self.mode=='test':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]
            image = Image.open(self.root+'val_images_256/'+img_path).convert('RGB')
            img = self.transform(image)
            return img, target

    def __len__(self):
        if self.mode!='test':
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)


    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):

        train_labels = np.array(list(self.train_labels.values()))
        raw_cls_num_list = np.array([sum(train_labels == i) for i in range(cls_num)])
        raw_cls_num_sort = raw_cls_num_list.argsort()[::-1]

        img_max = max(raw_cls_num_list)
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

        new_img_num_per_cls = [0 for _ in range(cls_num)]
        for i in range(cls_num):
            j = raw_cls_num_sort[i]
            new_img_num_per_cls[j] = min(img_num_per_cls[i], raw_cls_num_list[j])

        return new_img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls, imb_file=None):
        if os.path.exists(imb_file):
            new_data = json.load(open(imb_file,"r"))
        else:
            new_data = []

            cls_idx = [[] for _ in range(50)]
            for i, img in enumerate(self.train_data):
                target = self.train_labels[img]
                cls_idx[target].append(i)

            classes = np.array(range(50))
            self.num_per_cls_dict = dict()
            for the_class, the_img_num in zip(classes, img_num_per_cls):
                self.num_per_cls_dict[the_class] = the_img_num
                idx = cls_idx[the_class]
                np.random.shuffle(idx)
                selec_idx = idx[:the_img_num]
                new_data.extend(self.train_data[selec_idx, ...])
            print ('saving imbalance data to %s ...' % imb_file)
            json.dump(new_data, open(imb_file, 'w'))

        self.new_train_data = new_data


class webvision_dataloader():
    def __init__(self, batch_size, num_class, num_workers, root_dir, log, imb_ratio=1):

        self.batch_size = batch_size
        self.num_class = num_class
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.imb_factor = imb_ratio

        self.transform_train = transforms.Compose([
                transforms.Resize(320),
                transforms.RandomResizedCrop(299),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ])
        self.transform_test = transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ])
        self.transform_imagenet = transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ])

    def run(self,mode,pred=[],prob=[],refine_labels=None, imb_factor=1):
        if mode=='warmup':
            all_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="all", num_class=self.num_class, imb_factor=self.imb_factor)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)
            return trainloader

        elif mode=='train':
            labeled_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="labeled",num_class=self.num_class,pred=pred,probability=prob,log=self.log, refine_labels=refine_labels, imb_factor=self.imb_factor)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)

            unlabeled_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled",num_class=self.num_class,pred=pred,log=self.log, imb_factor=self.imb_factor)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)
            return labeled_trainloader, unlabeled_trainloader

        elif mode=='test':
            test_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_test, mode='test', num_class=self.num_class)
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size*20,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            return test_loader

        elif mode=='eval_train':
            eval_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_test, mode='all', num_class=self.num_class, imb_factor=self.imb_factor)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size*20,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            return eval_loader

        elif mode=='imagenet':
            imagenet_val = imagenet_dataset(root_dir=self.root_dir, transform=self.transform_imagenet, num_class=self.num_class)
            imagenet_loader = DataLoader(
                dataset=imagenet_val,
                batch_size=self.batch_size*20,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            return imagenet_loader

