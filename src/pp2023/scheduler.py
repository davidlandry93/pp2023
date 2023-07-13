import torch.optim.lr_scheduler


def reduce_lr_plateau(optimizer, *args, steps_per_epoch=None, **kwargs):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, *args, **kwargs)


def one_cycle(optimizer, *args, **kwargs):
    return torch.optim.lr_scheduler.OneCycleLR(optimizer, *args, **kwargs)
