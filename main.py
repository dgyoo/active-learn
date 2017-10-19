import os
import sys
import argparse
import importlib

import torch
import torch.backends.cudnn as cudnn

import utils
import train, val
from actdb import ActiveDB

sys.path.append('./db')
sys.path.append('./task')

def main():

    # Define task.
    parser = argparse.ArgumentParser(description='Large-scale deep learning framework.')
    parser.add_argument('--task', metavar='NAME', type=str, required=True,
            help='specify a task name that defined in $ROOT/task/')
    arg = parser.parse_args(sys.argv[1:3])
    task = importlib.import_module(arg.task)

    # Get task-specific options and print.
    task_opt = task.Option()
    opt = task_opt.opt
    print('Options.')
    for k in sorted(vars(opt)):
        if not k.startswith('dst_dir'):
            print('  {0}: {1}'.format(k, opt.__dict__[k]))

    # Build db.
    dst_dir_db = os.path.join(opt.dst_dir, opt.db)
    dst_path_db = os.path.join(dst_dir_db, 'db.pth')
    try:
        db = torch.load(dst_path_db)
        print('DB loaded.')
    except:
        db_module = importlib.import_module(opt.db)
        print('Make train DB.')
        db_train = db_module.make_dataset_train(opt.db_root)
        print('Make val DB.')
        db_val = db_module.make_dataset_val(opt.db_root)
        print('Save DB.')
        db = {'train': db_train, 'val': db_val}
        os.makedirs(dst_dir_db, exist_ok=True)
        torch.save(db, dst_path_db)
    db_train = db['train']
    db_val = db['val']

    # Estimate input statistics.
    dst_path_input_stats = os.path.join(dst_dir_db, 'db_stats.pth')
    try:
        input_stats = torch.load(dst_path_input_stats)
        print('DB input stats loaded.')
    except:
        print('Estimate DB input stats.')
        batch_manager_train = task.BatchManagerTrain(db_train, opt)
        input_stats = batch_manager_train.estimate_input_stats()
        os.makedirs(dst_dir_db, exist_ok=True)
        torch.save(input_stats, dst_path_input_stats)

    # Set destimation model directory.
    dst_dir_model = os.path.join(dst_dir_db, opt.arch)
    if opt.start_from:
        assert opt.start_from.endswith('.pth.tar')
        dst_dir_model = opt.start_from[:-8]
    if task_opt.changes:
        dst_dir_model += ',' + task_opt.changes

    # Apply active learning step to source model path, destination model directory/path.
    start_from = opt.start_from
    start_from_db = None
    for stage in range(opt.stage):
        start_from = os.path.join(dst_dir_model, '{:03d}.pth.tar'.format(opt.num_epoch))
        start_from_db = os.path.join(dst_dir_model, 'db_active.pth')
        assert opt.num_epoch == len(utils.Logger(os.path.join(dst_dir_model, 'val.log'))), \
                'Finish training before the next active learning stage.'
        dst_dir_model = os.path.join(dst_dir_model,
                '{:03d},sampler={},stage={}'.format(opt.num_epoch, opt.sampler, stage + 1))
    dst_path_model = os.path.join(dst_dir_model, '{:03d}.pth.tar')
    print('Active learning stage {}.'.format(opt.stage))

    # Initialize model, criterion, optimizer.
    model = task.Model(opt)

    # Create loggers.
    logger_train = utils.Logger(os.path.join(dst_dir_model, 'train.log'))
    logger_val = utils.Logger(os.path.join(dst_dir_model, 'val.log'))
    assert len(logger_train) == len(logger_val)

    # If models trained before, update informations to resume training.
    best_perform = 0
    start_epoch = len(logger_train)
    if start_epoch > 0:
        best_perform = logger_val.max()
        start_from = dst_path_model.format(start_epoch)
        start_from_db = os.path.join(dst_dir_model, 'db_active.pth')
    if start_epoch == opt.num_epoch:
        print('All done.')
        return

    # Fetch previouse parameters from that to resume training.
    dst_path_db_active = os.path.join(dst_dir_model, 'db_active.pth')
    os.makedirs(dst_dir_model, exist_ok=True)
    if start_from:
        print('Load a model from that to resume training.\n'
                '({})'.format(start_from))
        checkpoint = torch.load(start_from)
        model.model.load_state_dict(checkpoint['state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer'])
        print('Load active DB.')
        data = torch.load(start_from_db)
        db_active = ActiveDB(
                data,
                db_val,
                task.BatchManagerTrain,
                task.BatchManagerVal,
                input_stats,
                model,
                opt,
                dst_dir_model)
        if start_epoch == 0:
            print('Increase active DB labels.')
            db_active.increase_labels()
            print('Save increased active DB.')
            torch.save(db_active.db, dst_path_db_active)
    else:
        print('Make initial active DB.')
        data = {'pairs': [], 'pool': db_train['pairs'], 'log': [[]]}
        db_active = ActiveDB(
                data,
                db_val,
                task.BatchManagerTrain,
                task.BatchManagerVal,
                input_stats,
                model,
                opt,
                dst_dir_model)
        print('Save initial active DB.')
        torch.save(db_active.db, dst_path_db_active)

    # Set training db.
    db_train = db_active.db

    # Create batch manager.
    batch_manager_train = task.BatchManagerTrain(db_train, opt)
    batch_manager_train.set_input_stats(input_stats)
    batch_manager_val = task.BatchManagerVal(db_val, opt)
    batch_manager_val.set_input_stats(input_stats)

    # Cache input data if necessary.
    if opt.cache_train_data:
        batch_manager_train.cache_data()
    if opt.cache_val_data:
        batch_manager_val.cache_data()

    # If evaluation mode, evaluate the model and exit.
    if opt.evaluate:
        return val.val(batch_manager_val, model)

    # Do the job.
    cudnn.benchmark = True
    os.makedirs(dst_dir_model, exist_ok=True)
    for epoch in range(start_epoch, opt.num_epoch):

        # Adjust learning rate before training.
        learn_rate = opt.learn_rate * (0.1 ** (epoch // opt.decay_epoch))
        for param_group in model.optimizer.param_groups:
            param_group['lr'] = learn_rate

        # Train.
        print('\nStart training at epoch {}.'.format(epoch + 1))
        train.train(batch_manager_train, model, logger_train, epoch + 1)

        # Val.
        print('\nStart validation at epoch {}.'.format(epoch + 1))
        perform = val.val(batch_manager_val, model, logger_val, epoch + 1)

        # Save model.
        print('\nSave this model.')
        data = {
            'opt': opt,
            'log_train': logger_train.read(),
            'log_val': logger_val.read(),
            'state_dict': model.model.state_dict(),
            'optimizer' : model.optimizer.state_dict()}
        torch.save(data, dst_path_model.format(epoch + 1))

        # Remove previous model.
        if epoch > 0:
            print('Remove the previous model.')
            os.system('rm {}'.format(dst_path_model.format(epoch)))

        # Backup the best model.
        if perform > best_perform:
            print('Backup this model as the best.')
            os.system('cp {} {}'.format(
                dst_path_model.format(epoch + 1),
                os.path.join(dst_dir_model, 'best.pth.tar')))
            best_perform = perform

if __name__ == '__main__':
    main()
    print('')
