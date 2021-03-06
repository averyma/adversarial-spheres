import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np
from scipy.stats import norm

AVOID_ZERO_DIV = 1e-12
FORCE_POSITIVE_WEIGHT = -1e-6

def get_norm(x, dim=1):
    norm = torch.norm(x, p=2, dim=dim, keepdim=True)
    return norm

def normalize(x):
    # x_norm = torch.norm(x, p = 2, dim = dim, keepdim = True)
    x_norm = get_norm(x)
    normalized_x = x / x_norm.clamp(min=AVOID_ZERO_DIV)
    return normalized_x

def make_perfect_model(model, r, device):
    W, w ,b = get_model_params(model)
    W_shape = W.shape

    perfect_W = torch.cat([torch.eye(W_shape[1]),
                           torch.zeros([W_shape[0] - W_shape[1],
                           W_shape[1]])], dim=0)
    list(model.parameters())[0].data = perfect_W
    list(model.parameters())[1].data = torch.tensor(1.).view(1, 1)
    list(model.parameters())[2].data = torch.tensor(-r).view(1)

    alpha_percentage = get_alpha_stats(model, r)
    good_alpha = alpha_percentage[0]
    if np.abs(good_alpha - 100.) > 1e-5:
        raise Exception("Failed to make a perfect model! Good alpha: {0:.2f}%".format(good_alpha))
    print("Good alpha: {0:.2f}%".format(good_alpha))

    return model.to(device)

def make_positive_wb(model):
#     list(model.parameters())[1].data.clamp_(max = FORCE_POSITIVE_WEIGHT)
    w = list(model.parameters())[1].detach()
    b = list(model.parameters())[2].detach()

    if b > 0:
        list(model.parameters())[2].data = (-b).detach()
    if w < 0:
        list(model.parameters())[2].data = (-w).detach()

def make_sphere(num_samples, dim, r, device):
    z = torch.randn([int(num_samples), dim])
    x = normalize(z).to(device)
    mask = torch.rand(int(num_samples)) > 0.5
    x[mask, :] = r * x[mask, :]
    y = mask.float().to(device)
    return x, y

def make_truemax(model, r, device):
    alpha, u, v = get_alpha(model, True)
   
    # d_1: diff bw 1/r**2 and those alphas below 1/r**2
    d_1 = 1/(r**2) - alpha[alpha < 1/(r**2)]
    # d_2: diff bw 1 and those alphas above 1
    d_2 = alpha[alpha > 1] - 1
   
    # ipdb.set_trace()
    if len(d_1) == 0 and len(d_2) == 0:
        # all alphas are in the right range, we generate a random datapoint
        x, y = make_sphere(num_samples=1, dim=len(alpha), r=r, device=device)
        y = y.view([])
    elif len(d_1) == 0 or d_1.max() <= (bool(len(d_2)) and d_2.max()):
        # d_1 is empty: all bad_alpha are larger than 1
        # OR max alpha error comes from alphas larger than 1
        # use direction corresponds to max alpha(svalue) to construct inner sphere example.
        x = v[:, 0].view(1, len(alpha)).to(device)
        x = normalize(x)
        y = torch.tensor(0., device=device)
    else: 
        # d_2 is empty: all bad_alpha are smaller than 1/r**2
        # OR max alpha error comes from alphas smaller than 1/r**2
        # use direction corresponds to min alpha(svalue) to construct outer sphere example.
        x = v[:,-1].view(1,len(alpha)).to(device)
        x = r * normalize(x)
        y = torch.tensor(1., device = device)
    return x, y

def get_model_params(model):
    W = list(model.parameters())[0]
    w = list(model.parameters())[1]
    b = list(model.parameters())[2]
    return W, w ,b

def get_alpha(model, uv = False):
    #TODO: I am getting the following errors while doing torch.svd
    #RuntimeError: svd_cuda: the updating process of SBDSDC did not converge (error: 14)
    #So as a temporary work-around, I copied W to cpu. But now im getting a similar error:
    #RuntimeError: svd_cuda: the updating process of SBDSDC did not converge (error: 14)
    #I need to look into this, possibly use the numpy svd?
    W, w ,b = get_model_params(model)

    # if uv:
        # u,s,v = np.linalg.svd(W.detach().cpu().numpy(), compute_uv = uv)
        # u = torch.tensor(u, device = W.device)
        # s = torch.tensor(s, device = W.device)
        # v = torch.tensor(v, device = W.device)
    # else:
        # s = np.linalg.svd(W.detach().cpu().numpy(), compute_uv = uv)
        # u = torch.zeros(1, device = W.device)
        # s = torch.tensor(s, device = W.device)
        # v = torch.zeros(1, device = W.device)
    
    # u,s,v = torch.svd(W, compute_uv = uv)
    u,s,v = torch.svd(W.cpu(), compute_uv=uv)
    u = torch.tensor(u, device=W.device)
    s = torch.tensor(s, device=W.device)
    v = torch.tensor(v, device=W.device)
    
    alpha = (w * s**2 / (-b)).view(-1)
    return alpha, u, v

def get_alpha_stats(model, r):
    alpha, _, _ = get_alpha(model, False)
    total_alpha = alpha.numel()
    good_alpha = ((alpha >= 1/(r**2)) & (alpha <= 1)).sum().item()
    bad_alpha_inner = (alpha > 1).sum().item()
    bad_alpha_outer = (alpha < 1/(r**2)).sum().item()

    good_alpha /= (total_alpha/100)
    bad_alpha_inner /= (total_alpha/100)
    bad_alpha_outer /= (total_alpha/100)

    worst_err, avg_err = get_alpha_err(alpha, r)

    alpha_stats = [good_alpha, bad_alpha_inner, bad_alpha_outer, worst_err, avg_err]
    return alpha_stats

def get_alpha_err(alpha, r):

    # d_1: diff bw 1/r**2 and those alphas below 1/r**2
    d_1 = 1/(r**2) - alpha[alpha < 1/(r**2)]
    # d_2: diff bw 1 and those alphas above 1
    d_2 = alpha[alpha > 1] - 1

    # worst_error = torch.max(torch.max(alpha - 1).clamp(min = 0) + torch.max(1/r**2 - alpha).clamp(min = 0)
    if len(d_1) != 0 or len(d_2) != 0:
        worst_err = torch.cat([d_1, d_2]).max().item()
        avg_err = torch.cat([d_1, d_2]).mean().item()
    else:
        worst_err = 0
        avg_err = 0

    return worst_err, avg_err

def analytic_err(model, r):
    alpha, _, _ = get_alpha(model, False)

    # error rate based on inner sphere:
    mu_inner = (alpha - 1).sum().item()
    sigma_inner = np.sqrt(2*((alpha - 1)**2).sum().item())
    err_inner = 1 - norm.cdf(-mu_inner/sigma_inner)

    # error rate based on outer sphere:
    mu_outer = (1/r**2 - alpha).sum().item()
    sigma_outer = np.sqrt(2*((1/r**2 - alpha)**2).sum().item())
    err_outer = 1 - norm.cdf(-mu_outer/sigma_outer)

    err_avg = (err_inner + err_outer) * 0.5
    ana_err = [err_avg, err_inner, err_outer]

    return ana_err
