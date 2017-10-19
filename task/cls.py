import sys
import argparse
import os.path
from PIL import Image

import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torchvision.transforms as t
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import utils
import metric

##########################
# Task-specific options. #
##########################
class Option(object):

    def __init__(self):
        parser = argparse.ArgumentParser(description='Large-scale image classification')
        parser.add_argument('--db', default='imagenet', metavar='NAME', type=str,
                            help='dataset name')
        parser.add_argument('--db-root', default='/home/dgyoo/workspace/datain/ILSVRC', metavar='DIR', type=str,
                            help='root to the dataset')
        parser.add_argument('--dst-dir', default='/home/dgyoo/workspace/dataout/active-learn', metavar='DIR', type=str,
                            help='destination directory in that to save output data')
        parser.add_argument('--arch', default='resnet18', metavar='NAME', type=str,
                            help='model architecture')
        parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                            help='if specified, use a pre-trained model')
        parser.add_argument('--start-from', default='', metavar='PATH', type=str,
                            help='path to a model from that to resume training')
        parser.add_argument('--num-worker', default=16, metavar='N', type=int,
                            help='number of workers that provide mini-batches')
        parser.add_argument('--cache-train-data', dest='cache_train_data', action='store_true',
                            help='if specified, load all the training samples on the memory')
        parser.add_argument('--cache-val-data', dest='cache_val_data', action='store_true',
                            help='if specified, load all the validation samples on the memory')
        parser.add_argument('--num-epoch', default=90, metavar='N', type=int,
                            help='number of total epochs to run')
        parser.add_argument('--batch-size', default=256, metavar='N', type=int,
                            help='mini-batch size')
        parser.add_argument('--learn-rate', default=0.1, metavar='LR', type=float,
                            help='initial learning rate')
        parser.add_argument('--decay-epoch', default=30, metavar='N', type=int,
                            help='number of epochs after that to decay learning rate')
        parser.add_argument('--momentum', default=0.9, metavar='M', type=float,
                            help='momentum')
        parser.add_argument('--weight-decay', default=1e-4, metavar='W', type=float,
                            help='weight decay')
        parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                            help='if specified, evaluate model on validation set')
        parser.add_argument('--stage', default=0, metavar='N', type=int,
                            help='active learning stage')
        parser.add_argument('--seed', default=0, metavar='N', type=int,
                            help='random seed to create initial active db')
        parser.add_argument('--init-db-size', default=10000, metavar='N', type=int,
                            help='inieial labeled db size')
        parser.add_argument('--pool-size', default=100000, metavar='N', type=int,
                            help='unlabeled pool size (0: the rest)')
        parser.add_argument('--sampling-size', default=10000, metavar='N', type=int,
                            help='number of labeled samples added for each step')
        parser.add_argument('--sampler', default='random', metavar='NAME', type=str,
                            help='sampling strategy to add labeled samples')
        args = parser.parse_args(sys.argv[3:])
        ignore = ['db', 'db_root', 'dst_dir', 'arch', 'start_from', 'num_worker',
                'cache_train_data', 'cache_val_data', 'num_epoch', 'evaluate', 'stage', 'sampler']
        changes = utils.arg_changes(parser, args, ignore)
        self._opt = args
        self._changes = changes

    @property
    def opt(self):
        return self._opt

    @property
    def changes(self):
        return self._changes

############################
# Class to create a model. #
############################
class Model(object):

    def __init__(self, opt):
        if opt.pretrained:
            print('Load ' + opt.arch + ' pre-trained on image-net.')
            model = models.__dict__[opt.arch](pretrained=True)
        else:
            print('Create ' + opt.arch + '.')
            model = models.__dict__[opt.arch]()
        if opt.arch.startswith('alexnet') or opt.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
        else:
            model = torch.nn.DataParallel(model)
        self._model = model.cuda()
        self._criterion = nn.CrossEntropyLoss().cuda()
        self._optimizer = torch.optim.SGD(
                self.model.parameters(),
                opt.learn_rate,
                momentum=opt.momentum,
                weight_decay=opt.weight_decay)
    @property
    def model(self):
        return self._model

    @property
    def criterion(self):
        return self._criterion

    @property
    def optimizer(self):
        return self._optimizer

###############################
# Classes to provide batches. #
###############################
class BatchManagerTrain(Dataset):

    def __init__(self, db, opt, input_stats=None):
        self._db = db
        self._opt = opt
        self._input_stats = input_stats
        self._loader = _data_loader(self, opt.num_worker)
        self._evaluator = _evaluator
        self._index_perm = torch.randperm(len(db['pairs']))
        self._data_cache = None

    # Get batch.
    def __getitem__(self, index):

        images, targets, indices = [], [], []
        db_size = len(self._db['pairs'])
        for batch_index in range(self._opt.batch_size):

            # Determine a sample index.
            db_index = index * self._opt.batch_size + batch_index
            if db_index == db_size: break
            db_index = self._index_perm[db_index]

            # Processing from scratch.
            if self._data_cache is None:
                path, target = self._db['pairs'][db_index]
                with open(path, 'rb') as f:
                    with Image.open(f) as image:
                        image = image.convert('RGB')
                image = t.RandomSizedCrop(224)(image)
                image = t.RandomHorizontalFlip()(image)
                image = t.ToTensor()(image)
                if not self._input_stats is None:
                    image = t.Normalize(
                            mean=self._input_stats['mean'],
                            std=self._input_stats['std'])(image)

            # Loading from cache.
            else:
                image, target = self._data_cache[db_index]

            # Append.
            images.append(image)
            targets.append(target)
            indices.append(db_index)
        return default_collate(images), default_collate(targets), default_collate(indices)

    # Number of batches.
    def __len__(self):
        return -(-len(self._db['pairs']) // self._opt.batch_size)

    def estimate_input_stats(self, num_sample=1e5):
        num_sample = min(num_sample, len(self._db['pairs']))
        num_batch = int(-(-num_sample // self._opt.batch_size))
        mean_meter = utils.AverageMeter()
        std_meter = utils.AverageMeter()
        for i, (batch, _, _) in enumerate(self._loader):
            mean = batch.mean(3).mean(2).mean(0)
            std = batch.view(self._opt.batch_size, 3, -1).std(2).mean(0)
            mean_meter.update(mean, batch.size(0))
            std_meter.update(std, batch.size(0))
            print('Batch {}/{}, mean {}, std {}'.format(
                i + 1, num_batch,
                utils.to_string(mean_meter.avg, '%.4f'),
                utils.to_string(std_meter.avg, '%.4f')))
            if i + 1 == num_batch: break
        return {'mean': mean_meter.avg, 'std':std_meter.avg}

    def set_input_stats(self, input_stats):
        assert self._input_stats is None
        self._input_stats = input_stats

    def cache_data(self):
        assert not self._input_stats is None
        data_cache = [[] for _ in range(len(self._db['pairs']))]
        batch_manager = self.__class__(self._db, self._opt, self._input_stats)
        for i, (images, targets, indices) in enumerate(batch_manager.loader):
            for j in range(len(indices)):
                data_cache[indices[j]] = (images[j], targets[j])
            print('Batch {}/{}, cache data.'.format(i + 1, len(batch_manager)))
        self._data_cache = data_cache

    @property
    def loader(self):
        self._index_perm = torch.randperm(len(self._db['pairs']))
        return self._loader

    @property
    def evaluator(self):
        return self._evaluator

class BatchManagerVal(BatchManagerTrain):

    # Get batch.
    def __getitem__(self, index):

        images, targets, indices = [], [], []
        db_size = len(self._db['pairs'])
        for batch_index in range(self._opt.batch_size):

            # Determine a sample index.
            db_index = index * self._opt.batch_size + batch_index
            if db_index == db_size: break

            # Processing from scratch
            if self._data_cache is None:
                path, target = self._db['pairs'][db_index]
                with open(path, 'rb') as f:
                    with Image.open(f) as image:
                        image = image.convert('RGB')
                image = t.Scale(256)(image)
                image = t.CenterCrop(224)(image)
                image = t.ToTensor()(image)
                image = t.Normalize(
                        mean=self._input_stats['mean'],
                        std=self._input_stats['std'])(image)

            # Loading from cache.
            else:
                image, target = self._data_cache[db_index]

            # Append.
            images.append(image)
            targets.append(target)
            indices.append(db_index)

        return default_collate(images), default_collate(targets), default_collate(indices)

def _data_loader(batch_manager, num_worker):
    return DataLoader(
            batch_manager,
            batch_size=1,
            shuffle=False,
            sampler=None,
            batch_sampler=None,
            num_workers=num_worker,
            collate_fn=lambda x:x[0],
            pin_memory=True,
            drop_last=False)

def _evaluator(outputs, targets):
    return metric.accuracy(outputs, targets, topk=(1, 5))
