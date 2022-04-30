import torch

# L1 normalization
def l1norm(x, dim, eps=1e-8):
    norm = torch.abs(x).sum(dim=dim, keepdim=True) + eps
    x = torch.div(x, norm)
    return x

# L2 normalization
def l2norm(x, dim, eps=1e-8):
    norm = torch.pow(x,2).sum(dim=dim, keepdim=True).sqrt() + eps
    x = torch.div(x, norm)
    return x

# cosine similiarity
def cosine_sim(x, y, dim=1, eps=1e-8):
    '''
        x = [a, b, d]
        y = [a, b, d]
    '''
    # w12 = [a, b]
    w12 = torch.sum(x*y, dim)
    w1 = torch.norm(x, 2, dim)
    w2 = torch.norm(y, 2, dim)
    return (w12/(w1*w2).clamp(min=eps)).squeeze()