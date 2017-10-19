import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import metric
import train, val

class ActiveDB(object):

    def __init__(self, db, db_val, BatchManagerTrain, BatchManagerVal, input_stats, model, opt, dst_dir_agent):

        for k in db.keys(): assert k in ['pairs', 'pool', 'log']
        self._db = deepcopy(db)
        self._db_val = db_val
        self._BatchManagerTrain = BatchManagerTrain
        self._BatchManagerVal = BatchManagerVal
        self._input_stats = input_stats
        self._model = model
        self._opt = opt
        self._dst_dir_agent = dst_dir_agent

        # Build initial labeled db if necessary.
        if not self._db['pairs']:
            assert not self._db['log'][0]
            torch.manual_seed(opt.seed)
            pool = self._db['pool']
            self._db['pool'] = []
            indices = torch.randperm(len(pool)).tolist()
            for i in range(opt.init_db_size):
                self._db['pairs'].append(pool[indices[i]])
                self._db['log'][0].append(pool[indices[i]])
            pool_size = len(indices) - opt.init_db_size if opt.pool_size == 0 else opt.pool_size
            for i in range(pool_size):
                self._db['pool'].append(pool[indices[opt.init_db_size + i]])

    def _random_sampler(self):

        perm = torch.randperm(len(self._db['pool']))
        return perm[:self._opt.sampling_size]

    def _confidence_sampler(self):

        # Predict class posteriors over unlabeled set.
        posteriors, _, db_indices = self._compute_posteriors(self._db['pool'], self._model)

        # Compute confidence.
        confidences = [posterior.max() for posterior in posteriors]
        _, indices = torch.sort(torch.Tensor(confidences), descending=False)

        return torch.LongTensor(db_indices)[indices[:self._opt.sampling_size]]

    def _entropy_sampler(self):

        # Predict class posteriors over unlabeled set.
        posteriors, _, db_indices = self._compute_posteriors(self._db['pool'], self._model)

        # Compute entropies.
        entropies = []
        for posterior in posteriors:
            entropies.append(-torch.sum(torch.mul(torch.log(posterior), posterior)))
        _, indices = torch.sort(torch.Tensor(entropies), descending=True)

        return torch.LongTensor(db_indices)[indices[:self._opt.sampling_size]]

    def _ideal_sampler(self):

        # Predict class posteriors over unlabeled set.
        posteriors, targets, db_indices = self._compute_posteriors(self._db['pool'], self._model)

        # Compute softmax losses.
        losses = []
        for i, posterior in enumerate(posteriors):
            losses.append(-posterior[0, targets[i][0]])
        _, indices = torch.sort(torch.Tensor(losses), descending=True)

        return torch.LongTensor(db_indices)[indices[:self._opt.sampling_size]]

    def _agent_sampler(self, mode):
        # Define fuctions.
        if mode == 'reg':

            def _gen_agent_target(posteriors, targets):
                return [0.5 - posterior[0, targets[i][0]] for i, posterior in enumerate(posteriors)]

            def is_best(eval_best, eval_current):
                return eval_best > eval_current

            num_out_dim = 1
            type_fun = float
            criterion = nn.MSELoss()
            evaluator = metric.mse
            learn_rate = 0.001

        elif mode == 'cls':

            def _gen_agent_target(posteriors, targets):
                return [posterior[0, targets[i][0]] != posterior.max() for i, posterior in enumerate(posteriors)]
            
            def ap(outputs, targets):
                return metric.ap(F.softmax(outputs)[:,[1]].data, targets)

            def is_best(eval_best, eval_current):
                return eval_best < eval_current

            num_out_dim = 2
            type_fun = int
            criterion = nn.CrossEntropyLoss()
            evaluator = ap
            learn_rate = 0.01

        def _make_agent_dataset(db, db_indices, agent_targets, targets):
            pairs = []
            for i, agent_target in enumerate(agent_targets):
                image, target = db['pairs'][db_indices[i]]
                assert target == targets[i][0]
                pairs.append((image, type_fun(agent_target)))
            return {'pairs': pairs}

        # Predict class posteriors to compute agent targets.
        posteriors_train, targets_train, db_indices_train = self._compute_posteriors(self._db['pairs'], self._model)
        posteriors_val, targets_val, db_indices_val = self._compute_posteriors(self._db_val['pairs'], self._model)

        # Generate agent targets.
        agent_targets_train = _gen_agent_target(posteriors_train, targets_train)
        agent_targets_val = _gen_agent_target(posteriors_val, targets_val)

        # Create input-target pairs to learn an agent.
        agent_db_train = _make_agent_dataset(self._db, db_indices_train, agent_targets_train, targets_train)
        agent_db_val = _make_agent_dataset(self._db_val, db_indices_val, agent_targets_val, targets_val)

        # Create batch managers to learn an agent.
        batch_manager_train = self._BatchManagerTrain(agent_db_train, self._opt, self._input_stats)
        batch_manager_train._evaluator = evaluator
        batch_manager_val = self._BatchManagerVal(agent_db_val, self._opt, self._input_stats)
        batch_manager_val._evaluator = evaluator
        
        # Cache input data if necessary.
        if self._opt.cache_train_data:
            batch_manager_train.cache_data()
        if self._opt.cache_val_data:
            batch_manager_val.cache_data()

        # Create loggers.
        logger_train = utils.Logger(os.path.join(self._dst_dir_agent, 'agent-train.log'))
        logger_val = utils.Logger(os.path.join(self._dst_dir_agent, 'agent-val.log'))
        assert len(logger_train) == 0 and len(logger_val) == 0

        # Initialize an agent from the model.
        model = deepcopy(self._model.model.module)
        model.fc = nn.Linear(model.fc.in_features, num_out_dim)
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        optimizer = torch.optim.SGD(
                model.parameters(),
                self._opt.learn_rate,
                momentum=self._opt.momentum,
                weight_decay=self._opt.weight_decay)
        class Model(object):
            def __init__(self):
                self.model = model
                self.criterion = criterion.cuda()
                self.optimizer = optimizer
        agent = Model()

        # Learn the agent.
        best_perform = 0
        for param_group in agent.optimizer.param_groups:
                param_group['lr'] = learn_rate
        for epoch in range(3):
            print('\nStart agent training at epoch {}.'.format(epoch + 1))
            train.train(batch_manager_train, agent, logger_train, epoch + 1)
            print('\nStart agent validation at epoch {}.'.format(epoch + 1))
            perform = val.val(batch_manager_val, agent, logger_val, epoch + 1)
            if is_best(best_perform, perform):
                best_perform = perform
                best_agent = deepcopy(agent)

        # Predict uncertainty with the agent over unlabeled set.
        posteriors, _, db_indices = self._compute_posteriors(self._db['pool'], best_agent)
        
        # Compute uncertainties.
        uncertainties = []
        for i, posterior in enumerate(posteriors):
            uncertainties.append(posterior[0, 1])
        _, indices = torch.sort(torch.Tensor(uncertainties), descending=True)
        
        return torch.LongTensor(db_indices)[indices[:self._opt.sampling_size]]

    def _compute_posteriors(self, db, model):

        posteriors, targets, db_indices = [], [], []
        batch_manager = self._BatchManagerVal({'pairs':db}, self._opt)
        batch_manager.set_input_stats(self._input_stats)
        model.model.eval()
        for i, (input_batch, target_batch, index_batch) in enumerate(batch_manager.loader):
            output_batch = model.model(torch.autograd.Variable(input_batch, volatile=True))
            posterior_batch = F.softmax(output_batch)
            posteriors += list(posterior_batch.data.cpu().split(1, 0))
            targets += list(target_batch.split(1, 0))
            db_indices += index_batch.tolist()
            print('Batch {}/{}, compute class posteriors.'.format(i + 1, len(batch_manager)))
        return posteriors, targets, db_indices

    def increase_labels(self):

        get_sampler = {
                'random': self._random_sampler,
                'conf': self._confidence_sampler,
                'entropy': self._entropy_sampler,
                'reg-agent': lambda: self._agent_sampler('reg'),
                'cls-agent': lambda: self._agent_sampler('cls'),
                'ideal': self._ideal_sampler}
        indices = get_sampler[self._opt.sampler]()
        indices, _ = torch.sort(indices, descending=True, dim=0)
        to_add = [self._db['pool'].pop(index) for index in indices]
        self._db['pairs'] += to_add
        self._db['log'] += [to_add]

    @property
    def db(self):
        return self._db
