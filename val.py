import torch
import time
from collections import Iterable

import utils

def val(batch_manager, model, logger=None, epoch=0):
    # Initialize meters.
    data_time = utils.AverageMeter()
    net_time = utils.AverageMeter()
    loss_meter = utils.AverageMeter()
    eval_meter = utils.AverageMeter()

    # Do the job.
    loader = batch_manager.loader
    model.model.eval()
    t0 = time.time()
    for i, (inputs, targets, _) in enumerate(loader):
        # Set variables.
        if 'DoubleTensor' in targets.type():
            targets = targets.float()
        targets = targets.cuda(async=True)
        inputs_var = torch.autograd.Variable(inputs, volatile=True)
        targets_var = torch.autograd.Variable(targets, volatile=True)

        # Measure data time.
        data_time.update(time.time() - t0)
        t0 = time.time()

        # Forward.
        outputs = model.model(inputs_var)
        loss = model.criterion(outputs, targets_var)
        evals = batch_manager.evaluator(outputs.data, targets)

        # Accumulate statistics.
        loss_meter.update(loss.data[0], targets.size(0))
        eval_meter.update(evals, targets.size(0))

        # Measure network time.
        net_time.update(time.time() - t0)
        t0 = time.time()

        # Print iteration.
        print('Epoch {0} Batch {1}/{2} '
                'T-data {data_time.val:.2f} ({data_time.avg:.2f}) '
                'T-net {net_time.val:.2f} ({net_time.avg:.2f}) '
                'Loss {loss.val:.2f} ({loss.avg:.2f}) '
                'Eval {eval_val} ({eval_avg})'.format(
                    epoch, i + 1, len(loader),
                    data_time=data_time,
                    net_time=net_time,
                    loss=loss_meter,
                    eval_val=utils.to_string(eval_meter.val),
                    eval_avg=utils.to_string(eval_meter.avg)))

    # Summerize results.
    perform = eval_meter.avg
    if not isinstance(perform, Iterable): perform = [perform]
    if not logger is None: logger.write([epoch, loss_meter.avg] + perform)
    print('Summary of validation at epoch {epoch:d}.\n'
            '  Number of sample: {num_sample:d}\n'
            '  Number of batches: {num_batch:d}\n'
            '  Total time for data: {data_time:.2f} sec\n'
            '  Total time for network: {net_time:.2f} sec\n'
            '  Total time: {total_time:.2f} sec\n'
            '  Average loss: {avg_loss:.4f}\n'
            '  Performance: {avg_perf}'.format(
                epoch=epoch,
                num_sample=loss_meter.count,
                num_batch=len(loader),
                data_time=data_time.sum,
                net_time=net_time.sum,
                total_time=data_time.sum+net_time.sum,
                avg_loss=loss_meter.avg,
                avg_perf=utils.to_string(eval_meter.avg, '%.4f')))

    return perform[0]
