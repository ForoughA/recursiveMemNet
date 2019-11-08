from torch.optim import SGD
from torch.optim import Adam

def build_optimizer(params, optimizer_type, lr, momentum, weight_decay, beta1, beta2):
    if optimizer_type == "sgd":
        return SGD(params, lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == "adam":
        return Adam(params, lr, weight_decay=weight_decay, betas=(beta1, beta2))
    else:
        raise ValueError("Unhandled optimizer type:%s" % optimizer_type)