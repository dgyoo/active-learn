import torch

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size)[0])
    return res

def mse(output, target):
    return torch.mean(torch.pow(output - target, 2))

def ap(output, target):
    assert output.numel() == output.size(0)
    assert target.numel() == target.size(0)
    assert output.numel() == target.numel()
    assert target.numel() == target.sum() + (1 - target).sum()
    output = output.cpu()
    target = target.cpu().float()
    num_positive = target.sum()
    _, rank_to_index = torch.sort(output, descending=True, dim=0)
    rank_to_target = target[rank_to_index.squeeze()]
    true_positive = torch.cat((torch.zeros(1), rank_to_target.cumsum(0)), 0)
    false_positive = torch.cat((torch.zeros(1), (1 - rank_to_target).cumsum(0)), 0)
    recall = true_positive / num_positive
    precision = true_positive / torch.clamp(true_positive + false_positive, min=1e-8)
    positive_indices = torch.cat((torch.zeros(1).byte(), (recall[1:] - recall[:-1]).gt(0)), 0).squeeze()
    return precision[positive_indices].sum() / num_positive
