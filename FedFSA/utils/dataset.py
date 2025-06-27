import numpy as np
import torchvision
from torchvision import transforms
import torch
from torch.utils import data
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import ssl
import sys
import random
ssl._create_default_https_context = ssl._create_unverified_context
from PIL import Image

class TinyImageNet_load(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")
        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))
        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(self.train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt

class DatasetObject:
    def __init__(self, dataset, n_client, seed, rule, class_main, frac_data=0.7, dir_alpha=0):
        self.dataset = dataset
        self.n_client = n_client
        self.seed = seed
        self.rule = rule
        self.frac_data = frac_data
        self.dir_alpha = dir_alpha
        self.class_main = class_main

        self.name = "Data%s_nclient%d_seed%d_rule%s_alpha%s_class_main%d_frac_data%s" % (
        self.dataset, self.n_client, self.seed, self.rule, self.dir_alpha, self.class_main, self.frac_data)
        self.set_data()

    def set_data(self):
        # Prepare data if not ready
        if not os.path.exists('Data/%s' % (self.name)):

            if self.dataset == 'FMNIST':
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
                trnset = torchvision.datasets.FashionMNIST(root='Data/', train=True, download=True, transform=transform)
                tstset = torchvision.datasets.FashionMNIST(root='Data/', train=False, download=True,
                                                            transform=transform)

                trn_load = torch.utils.data.DataLoader(trnset, batch_size=60000, shuffle=False, num_workers=1)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 1
                self.width = 28
                self.height = 28
                self.n_cls = 10

            if self.dataset == 'CIFAR10':
                transform = transforms.Compose([transforms.ToTensor()])

                trnset = torchvision.datasets.CIFAR10(root='Data/', train=True, download=True, transform=transform)
                tstset = torchvision.datasets.CIFAR10(root='Data/', train=False, download=True, transform=transform)
                trn_load = torch.utils.data.DataLoader(trnset, batch_size=50000, shuffle=False, num_workers=1)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)

                self.channels = 3
                self.width = 32
                self.height = 32
                self.n_cls = 10

            if self.dataset == 'CIFAR100':
                transform = transforms.Compose([transforms.ToTensor()])

                trnset = torchvision.datasets.CIFAR100(root='Data/', train=True, download=True, transform=transform)
                tstset = torchvision.datasets.CIFAR100(root='Data/', train=False, download=True, transform=transform)

                trn_load = torch.utils.data.DataLoader(trnset, batch_size=50000, shuffle=False, num_workers=1)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 3
                self.width = 32
                self.height = 32
                self.n_cls = 100
                # Get Raw data
            if self.dataset == 'TINY':
                if not os.path.exists(f'./Data/tiny-imagenet-200.zip'):
                    os.system(
                        f'wget --directory-prefix ./Data http://cs231n.stanford.edu/tiny-imagenet-200.zip')
                    os.system(f'unzip -q ./Data/tiny-imagenet-200.zip -d ./Data/')
                else:
                    print('rawdata TINY already exists.\n')
                transform = transforms.Compose(
                    [transforms.ToTensor()])
                trnset = TinyImageNet_load('./Data/tiny-imagenet-200/', train=True, transform=transform)
                tstset = TinyImageNet_load('./Data/tiny-imagenet-200/', train=False, transform=transform)
                trn_load = torch.utils.data.DataLoader(
                    trnset, batch_size=len(trnset), shuffle=False, num_workers=1)
                tst_load = torch.utils.data.DataLoader(
                    tstset, batch_size=len(tstset), shuffle=False, num_workers=1)
                self.channels = 3
                self.width = 64
                self.height = 64
                self.n_cls = 200

            if self.dataset == 'FMNIST' or self.dataset == 'CIFAR10' or self.dataset == 'CIFAR100' or self.dataset == 'TINY':
                trn_itr = trn_load.__iter__()
                tst_itr = tst_load.__iter__()

                trn_x, trn_y = trn_itr.__next__()
                tst_x, tst_y = tst_itr.__next__()

                trn_x = trn_x.numpy()
                trn_y = trn_y.numpy().reshape(-1, 1)
                tst_x = tst_x.numpy()
                tst_y = tst_y.numpy().reshape(-1, 1)

                concat_datasets_x = np.concatenate((trn_x, tst_x), axis=0)
                concat_datasets_y = np.concatenate((trn_y, tst_y), axis=0)

                self.trn_x = trn_x
                self.trn_y = trn_y
                self.tst_x = tst_x
                self.tst_y = tst_y

            # Shuffle Data
            np.random.seed(self.seed)
            rand_perm = np.random.permutation(len(concat_datasets_y))

            concat_datasets_x = concat_datasets_x[rand_perm]
            concat_datasets_y = concat_datasets_y[rand_perm]

            assert len(concat_datasets_y) % self.n_client == 0
            n_data_per_clnt = int((len(concat_datasets_y)) / self.n_client)
            clnt_data_list = np.ones(self.n_client).astype(int) * n_data_per_clnt

            n_data_per_clnt_train = int(n_data_per_clnt * self.frac_data)
            n_data_per_clnt_tst = n_data_per_clnt - n_data_per_clnt_train
            clnt_data_list_train = np.ones(self.n_client).astype(int) * n_data_per_clnt_train
            clnt_data_list_tst = np.ones(self.n_client).astype(int) * n_data_per_clnt_tst

            cls_per_client = self.class_main

            cls_per_client_nums = n_data_per_clnt/cls_per_client
            n_cls = self.n_cls
            n_client = self.n_client
            
            # Distribute training datapoints
            idx_list = [np.where(concat_datasets_y == i)[0] for i in range(self.n_cls)]
            cls_amount = np.asarray([len(idx_list[i]) for i in range(self.n_cls)])

            if self.rule == 'Dirichlet':
                # Generate a list of CLIENT*CLASS by sampling from Dirichlet distribution
                cls_priors = np.random.dirichlet(alpha=[self.dir_alpha] * self.n_cls, size=self.n_client)
                prior_cumsum = np.cumsum(cls_priors, axis=1)
                concat_clnt_x = np.asarray(
                    [np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32) for
                    clnt__ in range(self.n_client)])
                concat_clnt_y = np.asarray(
                    [np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client)])
                # All client data quotas have been allocated, stop
                clients_list = list(range(self.n_client))
                # Flag indicating whether the category data has been extracted
                class_done = [False] * len(cls_amount)

                while (np.sum(clnt_data_list) != 0):
                    # Randomly select a client to assign data
                    curr_clnt = random.choice(clients_list)
                    # When there is only one client left, all the data will be given to it.
                    if len(clients_list) == 1:
                        print(f"last client:{clients_list[0]}")
                        break
                    # If current client is full resample a client
                    if clnt_data_list[curr_clnt] <= 0:
                        clients_list.remove(curr_clnt)
                        continue
                    clnt_data_list[curr_clnt] -= 1
                    curr_prior = prior_cumsum[curr_clnt]
                    while True:
                        cls_label = np.argmax(np.random.uniform(low = curr_prior[0],high = curr_prior[len(curr_prior)-1]) <= curr_prior)
                        if class_done[cls_label]==True:
                            continue
                        # If the class (cls_label) data has been allocated, update the probability distribution of all clients
                        if cls_amount[cls_label] <= 0:
                            class_done[cls_label]=True
                            # Go through each client and set the probability of a specific class to 0
                            for i in range(len(cls_priors)):  
                                cls_priors[i][cls_label] = 0 
                            prior_cumsum = np.cumsum(cls_priors, axis=1)
                            curr_prior = prior_cumsum[curr_clnt]
                            print(f"class:{cls_label} done")
                            continue
                        cls_amount[cls_label] -= 1

                        concat_clnt_x[curr_clnt][clnt_data_list[curr_clnt]] = concat_datasets_x[idx_list[cls_label][cls_amount[cls_label]]]
                        concat_clnt_y[curr_clnt][clnt_data_list[curr_clnt]] = concat_datasets_y[idx_list[cls_label][cls_amount[cls_label]]]

                        break
                curr_clnt = clients_list[0]
                # Give all remaining data to the last client with a balance
                while(clnt_data_list[curr_clnt]!=0):
                    clnt_data_list[curr_clnt] -= 1
                    for i in range(len(cls_amount)):
                        if cls_amount[i] != 0:
                            cls_amount[i] -= 1
                            concat_clnt_x[curr_clnt][clnt_data_list[curr_clnt]] = concat_datasets_x[idx_list[i][cls_amount[i]]]
                            concat_clnt_y[curr_clnt][clnt_data_list[curr_clnt]] = concat_datasets_y[idx_list[i][cls_amount[i]]]

                concat_clnt_x = np.asarray(concat_clnt_x)
                concat_clnt_y = np.asarray(concat_clnt_y)

                clnt_x = np.asarray(
                    [np.zeros((clnt_data_list_train[clnt__], self.channels, self.height, self.width)).astype(np.float32)
                        for
                        clnt__ in range(self.n_client)])
                clnt_y = np.asarray(
                    [np.zeros((clnt_data_list_train[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client)])
                tst_x = np.asarray(
                    [np.zeros((clnt_data_list_tst[clnt__], self.channels, self.height, self.width)).astype(np.float32)
                        for
                        clnt__ in range(self.n_client)])
                tst_y = np.asarray(
                    [np.zeros((clnt_data_list_tst[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client)])
                # The data allocated to each client is further divided into training set and test set
                for jj in range(n_client):
                    rand_perm = np.random.permutation(len(concat_clnt_y[jj]))
                    concat_clnt_x[jj] = concat_clnt_x[jj][rand_perm]
                    concat_clnt_y[jj] = concat_clnt_y[jj][rand_perm]

                    clnt_x[jj] = concat_clnt_x[jj][:n_data_per_clnt_train, :, :, :]
                    tst_x[jj] = concat_clnt_x[jj][n_data_per_clnt_train:, :, :, :]

                    clnt_y[jj] = concat_clnt_y[jj][:n_data_per_clnt_train, :]
                    tst_y[jj] = concat_clnt_y[jj][n_data_per_clnt_train:, :]


                cls_means = np.zeros((self.n_client, self.n_cls))
                for clnt in range(self.n_client):
                    for cls in range(self.n_cls):
                        cls_means[clnt,cls] = np.mean(clnt_y[clnt]==cls)
                prior_real_diff = np.abs(cls_means-cls_priors)
                print('--- Max deviation from prior: %.4f' %np.max(prior_real_diff))
                print('--- Min deviation from prior: %.4f' %np.min(prior_real_diff))

            if self.rule == 'noniid':
                shards_per_class = cls_per_client * n_client // n_cls
                assert cls_per_client * n_client % n_cls == 0, "Total number of shards of label is not evenly divisible."

                shard_size = len(concat_datasets_y) // (n_cls * shards_per_class)
                assert len(concat_datasets_y) %(n_cls * shards_per_class) == 0, "Total number of each class is not evenly divisible."
                # Create lists to store data for each client
                concat_clnt_x = np.asarray(
                    [np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32) for
                    clnt__ in range(self.n_client)])
                print(concat_clnt_x.shape)
                concat_clnt_y = np.asarray(
                    [np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client)])
                clients_list = list(range(self.n_client))
                cls_list = list(range(self.n_cls))
                for curr_clnt in clients_list:
                    cls_lists_temp = []
                    while(clnt_data_list[curr_clnt] > 0):
                        curr_cls = random.choice(cls_list)
                        all_in_cls_lists_temp = all(x in cls_lists_temp for x in cls_list)
                        if curr_clnt != len(clients_list)-1:
                            while curr_cls in cls_lists_temp :
                                if all_in_cls_lists_temp:
                                    break
                                curr_cls = random.choice(cls_list)
                        if curr_cls not in cls_lists_temp:
                            cls_lists_temp.append(curr_cls)
                        count = 0
                        while count<shard_size:
                            cls_amount[curr_cls] -= 1
                            clnt_data_list[curr_clnt] -= 1
                            concat_clnt_x[curr_clnt][clnt_data_list[curr_clnt]] = concat_datasets_x[idx_list[curr_cls][cls_amount[curr_cls]]]
                            concat_clnt_y[curr_clnt][clnt_data_list[curr_clnt]] = concat_datasets_y[idx_list[curr_cls][cls_amount[curr_cls]]]
                            count +=1

                        if cls_amount[curr_cls] <= 0:
                            cls_list.remove(curr_cls)

                concat_clnt_x = np.asarray(concat_clnt_x)
                concat_clnt_y = np.asarray(concat_clnt_y)
                
                clnt_x = np.asarray(
                    [np.zeros((clnt_data_list_train[clnt__], self.channels, self.height, self.width)).astype(np.float32)
                        for
                        clnt__ in range(self.n_client)])
                clnt_y = np.asarray(
                    [np.zeros((clnt_data_list_train[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client)])
                tst_x = np.asarray(
                    [np.zeros((clnt_data_list_tst[clnt__], self.channels, self.height, self.width)).astype(np.float32)
                        for
                        clnt__ in range(self.n_client)])
                tst_y = np.asarray(
                    [np.zeros((clnt_data_list_tst[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client)])
                # The data allocated to each client is further divided into training set and test set
                for jj in range(n_client):
                    rand_perm = np.random.permutation(len(concat_clnt_y[jj]))
                    concat_clnt_x[jj] = concat_clnt_x[jj][rand_perm]
                    concat_clnt_y[jj] = concat_clnt_y[jj][rand_perm]

                    clnt_x[jj] = concat_clnt_x[jj][:n_data_per_clnt_train, :, :, :]
                    tst_x[jj] = concat_clnt_x[jj][n_data_per_clnt_train:, :, :, :]

                    clnt_y[jj] = concat_clnt_y[jj][:n_data_per_clnt_train, :]
                    tst_y[jj] = concat_clnt_y[jj][n_data_per_clnt_train:, :]

            self.clnt_x = clnt_x
            self.clnt_y = clnt_y
            self.tst_x = tst_x
            self.tst_y = tst_y

            # Save data
            os.mkdir('Data/%s' % (self.name))

            np.save('Data/%s/clnt_x.npy' % (self.name), clnt_x)
            np.save('Data/%s/clnt_y.npy' % (self.name), clnt_y)

            np.save('Data/%s/tst_x.npy' % (self.name), tst_x)
            np.save('Data/%s/tst_y.npy' % (self.name), tst_y)

        else:
            print("Data is already downloaded")

            self.clnt_x = np.load('Data/%s/clnt_x.npy' % (self.name))
            self.clnt_y = np.load('Data/%s/clnt_y.npy' % (self.name))
            self.n_client = len(self.clnt_x)

            self.tst_x = np.load('Data/%s/tst_x.npy' % (self.name))
            self.tst_y = np.load('Data/%s/tst_y.npy' % (self.name))

            if self.dataset == 'FMNIST':
                self.channels = 1; self.width = 28; self.height = 28;self.n_cls = 10
            if self.dataset == 'CIFAR10':
                self.channels = 3; self.width = 32;self.height = 32;self.n_cls = 10
            if self.dataset == 'CIFAR100':
                self.channels = 3;self.width = 32;self.height = 32;self.n_cls = 100
            if self.dataset == 'TINY':
                self.channels = 3; self.width = 64; self.height = 64; self.n_cls = 200
        print('Class frequencies:')

        # train data
        count = 0
        for clnt in range(self.n_client):
            print(self.n_cls)
            print("Client %3d: " % clnt +
                    ', '.join(["%.3f" % np.mean(self.clnt_y[clnt] == cls) for cls in range(self.n_cls)]) +
                    ', Amount:%d' % self.clnt_y[clnt].shape[0])
            count += self.clnt_y[clnt].shape[0]

        print('Total Amount:%d' % count)
        print('-----------------------------------------------------------')
        # test data
        count = 0
        for clnt in range(self.n_client):
            print("Client %3d: " % clnt +
                    ', '.join(["%.3f" % np.mean(self.tst_y[clnt] == cls) for cls in range(self.n_cls)]) +
                    ', Amount:%d' % self.tst_y[clnt].shape[0])
            count += self.tst_y[clnt].shape[0]

        print('Total Amount:%d' % count)
        print('-----------------------------------------------------------')

# dataset
class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_x, data_y=True, train=False, dataset_name=''):
        self.name = dataset_name

        if self.name == 'FMNIST':
            self.X_data = torch.tensor(data_x).float()
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = torch.tensor(data_y).float()

        elif self.name == 'CIFAR10':
            self.train = train
            self.transform = transforms.Compose([transforms.ToTensor()])
            self.X_data = torch.tensor(data_x).float()
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = data_y.astype('float32')
            self.augment_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])])
            self.noaugmt_transform = transforms.Compose(
                [transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])])

        elif self.name == 'CIFAR100':
            self.train = train
            self.transform = transforms.Compose([transforms.ToTensor()])
            self.X_data = torch.tensor(data_x).float()
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = data_y.astype('float32')
            self.augment_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])])
            self.noaugmt_transform = transforms.Compose(
                [transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])])
        elif self.name == 'TINY':
            self.train = train
            self.transform = transforms.Compose([transforms.ToTensor()])
            self.X_data = torch.tensor(data_x).float()
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = data_y.astype('float32')
            self.augment_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            self.noaugmt_transform = transforms.Compose(
                [transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):

        if self.name == 'FMNIST':
            X = self.X_data[idx]
            if isinstance(self.y_data, bool):
                return X
            else:
                y = self.y_data[idx]
                return X, y

        elif self.name == 'CIFAR10':
            img = self.X_data[idx]
            if self.train:
                img = self.augment_transform(img)
            else:
                img = self.noaugmt_transform(img)
            if isinstance(self.y_data, bool):
                return img
            else:
                y = self.y_data[idx]
                return img, y

        elif self.name == 'CIFAR100':
            img = self.X_data[idx]
            if self.train:
                img = self.augment_transform(img)
            else:
                img = self.noaugmt_transform(img)
            if isinstance(self.y_data, bool):
                return img
            else:
                y = self.y_data[idx]
                return img, y
        elif self.name == 'TINY':
            img = self.X_data[idx]
            if self.train:
                img = self.augment_transform(img)
            else:
                img = self.noaugmt_transform(img)
            if isinstance(self.y_data, bool):
                return img
            else:
                y = self.y_data[idx]
                return img, y

if __name__ == '__main__':
    n_client = 100
    data_obj = DatasetObject(dataset='TINY', n_client=10, seed=23, rule = 'Dirichlet', class_main=15,
                            frac_data=0.7, dir_alpha=0.3)
    tst_x = data_obj.clnt_x
    tst_y = data_obj.clnt_y

    
    trn_gen = data.DataLoader(Dataset(tst_x[0], tst_y[0], train=True, dataset_name='TINY'), batch_size=48, shuffle=True)

    tst_gen_iter = trn_gen.__iter__()
    
    local_epoch = int(np.ceil(tst_x[0].shape[0] / 48))
    print("local_epoch")
    print(local_epoch)
    for i in range(local_epoch):
        data, target = tst_gen_iter.__next__()
        # print(target.shape)
        targets = target.reshape(-1)
        # print(targets.shape)
        # print(target[1])
        # print(target[2,0])
        print(target[1].type(torch.long))
        print(target[target[1].type(torch.long)])
        # print(target[:, None].shape)
        # print(target[:, None].expand(-1,-1, 5))
    '''   
    '''