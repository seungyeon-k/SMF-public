from torch.nn import MSELoss, CrossEntropyLoss
from loss.chamfer_loss import ChamferLoss

def get_loss(cfg, *args, device=None, version=None, **kwargs):
    name = cfg['type']
    loss = _get_loss_instance(name)
    return loss(**cfg)

def _get_loss_instance(name):
    try:
        return {
            "mse": get_mse_loss,
            "cross_entropy": get_ce_loss,
            "chamfer": get_chamfer_loss
        }[name]
    except:
        raise ("Loss {} not available".format(name))

def get_mse_loss(**kwargs):
    loss = MSELoss()
    return loss

def get_ce_loss(**kwargs):
    loss = CrossEntropyLoss()
    return loss

def get_chamfer_loss(**kwargs):
    loss = ChamferLoss()
    return loss