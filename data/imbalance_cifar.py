import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import json
import os

class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, noise_mode='sym', noise_ratio=0,
                 rand_number=0, train=True, transform=None, target_transform=None, download=False):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        random.seed(rand_number)
        
        if train:
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            self.gen_imbalanced_data(img_num_list)
            
            noise_file = os.path.join(root, 'cifar' + str(self.cls_num) + '_' + imb_type + '_' + str(imb_factor) + '_' + noise_mode + '_' + str(noise_ratio))
            self.get_noisy_data(self.cls_num, noise_file, noise_mode, noise_ratio)

        self.labels = self.targets

    def __getitem__(self, index):
        x, y = super(IMBALANCECIFAR10, self).__getitem__(index)
        return x, y, index

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

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
    
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

            elif noise_mode == 'new':
                r = 1 - sum([(n / num_train) ** 2 for n in cls_num_list])
                num_noise = int(noise_ratio / r * num_train)
                noise_idx = idx[:num_noise]

                p = np.array([cls_num_list for _ in range(cls_num)])
                p = p / p.sum(axis=1, keepdims=True)
                for i in range(num_train):
                    if i in noise_idx:
                        newlabel = np.random.choice(cls_num, p=p[train_label[i]])
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

class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = IMBALANCECIFAR100(root='./data', train=True,
                    download=True, transform=transform)
    trainloader = iter(trainset)
    data, label = next(trainloader)
    import pdb; pdb.set_trace()