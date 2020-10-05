import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
from torch.autograd import grad

from utils_sphere import get_alpha_stats, analytic_err, make_sphere, make_truemax
from attacks import pgd_sphere

AVOID_ZERO_DIV = 1e-12

def train_clean(logger, dim, r, total_samples, batch_size, err_freq, model, opt, device):
    num_itr = int(total_samples/batch_size)
    model.train()
    with trange(num_itr) as t:
        for i in range(num_itr):
            x, y = make_sphere(batch_size, dim, r, device)
            yp = model(x)

            loss = nn.BCEWithLogitsLoss(reduction="mean")(yp, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_correct = ((yp > 0).bool() == y.bool()).sum().item()
            batch_acc = batch_correct / batch_size * 100

            t.set_postfix(loss=loss.item(),
                          acc="{0:.2f}%".format(batch_acc),
                          good_alpha="nan" if not logger.log_dict["alpha/good"]
                          else "{0:.2f}%".format(logger.log_dict["alpha/good"][-1][2]))
            t.update()

            if (i+1) % err_freq == 0:
                test_acc, test_loss = test_clean(dim, r, 10000, 500, model, device)
                ana_err = analytic_err(model, r)
                alpha_stats = get_alpha_stats(model, r)

                logger.add_scalar("acc/test", test_acc, i+1)
                logger.add_scalar("acc/batch", batch_acc, i+1)
                logger.add_scalar("loss/test", test_loss, i+1)
                logger.add_scalar("loss/batch", loss.item(), i+1)
                logger.add_scalar("ana_err/avg", ana_err[0], i+1)
                logger.add_scalar("ana_err/inner", ana_err[1], i+1)
                logger.add_scalar("ana_err/outer", ana_err[2], i+1)
                logger.add_scalar("alpha/good", alpha_stats[0], i+1)
                logger.add_scalar("alpha/bad_inner", alpha_stats[1], i+1)
                logger.add_scalar("alpha/bad_outer", alpha_stats[2], i+1)
                logger.add_scalar("err_in_alpha/worst", alpha_stats[3], i+1)
                logger.add_scalar("err_in_alpha/avg", alpha_stats[4], i+1)

                if np.abs(alpha_stats[0] - 100.) < 1e-5:
                    break

    log = logger.log_dict
    return log

def train_truemax(logger, dim, r, total_iterations, err_freq, model, opt, device):
    num_itr = int(total_iterations)
    model.train()
    with trange(num_itr) as t:
        for i in range(num_itr):

            x, y = make_truemax(model, r, device)

            yp = model(x)
            loss = nn.BCEWithLogitsLoss(reduction="mean")(yp, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_correct = ((yp > 0).bool() == y.bool()).sum().item()
            batch_acc = batch_correct * 100

            t.set_postfix(loss=loss.item(),
                          acc="{0:.2f}%".format(batch_acc),
                          good_alpha="nan" if not logger.log_dict["alpha/good"]
                          else "{0:.2f}%".format(logger.log_dict["alpha/good"][-1][2]))
            t.update()

            if (i+1) % err_freq == 0:
                test_acc, test_loss = test_clean(dim, r, 10000, 500, model, device)
                ana_err = analytic_err(model, r)
                alpha_stats = get_alpha_stats(model, r)

                logger.add_scalar("acc/test", test_acc, i+1)
                logger.add_scalar("acc/batch", batch_acc, i+1)
                logger.add_scalar("loss/test", test_loss, i+1)
                logger.add_scalar("loss/batch", loss.item(), i+1)
                logger.add_scalar("ana_err/avg", ana_err[0], i+1)
                logger.add_scalar("ana_err/inner", ana_err[1], i+1)
                logger.add_scalar("ana_err/outer", ana_err[2], i+1)
                logger.add_scalar("alpha/good", alpha_stats[0], i+1)
                logger.add_scalar("alpha/bad_inner", alpha_stats[1], i+1)
                logger.add_scalar("alpha/bad_outer", alpha_stats[2], i+1)
                logger.add_scalar("err_in_alpha/worst", alpha_stats[3], i+1)
                logger.add_scalar("err_in_alpha/avg", alpha_stats[4], i+1)

                if np.abs(alpha_stats[0] - 100.) < 1e-5:
                    break

    log = logger.log_dict
    return log

def train_adv(logger, param, dim, r, total_samples, batch_size, err_freq, model, opt, device):
    attack_param = {'eps': param["eps"], 'num_iter': param["num_iter"], 'loss_fn': nn.BCEWithLogitsLoss()}
    num_itr = int(total_samples/batch_size)
    model.train()
    with trange(num_itr) as t:
        for i in range(num_itr):
            x, y = make_sphere(batch_size, dim, r, device)
            delta = pgd_sphere(**attack_param).generate(model, x, y)

            yp_clean = model(x)
            loss_clean = nn.BCEWithLogitsLoss()(yp_clean, y)
            yp_adv = model(x+delta)
            loss_adv = nn.BCEWithLogitsLoss()(yp_adv, y)

            loss = loss_clean + loss_adv

            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_correct = ((yp_clean > 0).bool() == y.bool()).sum().item()
            batch_acc = batch_correct / batch_size * 100

            t.set_postfix(loss=loss.item(),
                          acc="{0:.2f}%".format(batch_acc),
                          good_alpha="nan" if not logger.log_dict["alpha/good"]
                          else "{0:.2f}%".format(logger.log_dict["alpha/good"][-1][2]))
            t.update()

            if (i+1) % err_freq == 0:
                test_acc, test_loss = test_clean(dim, r, 10000, 500, model, device)
                ana_err = analytic_err(model, r)
                alpha_stats = get_alpha_stats(model, r)

                logger.add_scalar("acc/test", test_acc, i+1)
                logger.add_scalar("acc/batch", batch_acc, i+1)
                logger.add_scalar("loss/test", test_loss, i+1)
                logger.add_scalar("loss/batch", loss.item(), i+1)
                logger.add_scalar("ana_err/avg", ana_err[0], i+1)
                logger.add_scalar("ana_err/inner", ana_err[1], i+1)
                logger.add_scalar("ana_err/outer", ana_err[2], i+1)
                logger.add_scalar("alpha/good", alpha_stats[0], i+1)
                logger.add_scalar("alpha/bad_inner", alpha_stats[1], i+1)
                logger.add_scalar("alpha/bad_outer", alpha_stats[2], i+1)
                logger.add_scalar("err_in_alpha/worst", alpha_stats[3], i+1)
                logger.add_scalar("err_in_alpha/avg", alpha_stats[4], i+1)

                if np.abs(alpha_stats[0] - 100.) < 1e-5:
                    break

    log = logger.log_dict
    return log

def train_reg_1st(logger, lambbda, dim, r, total_samples, batch_size, err_freq, model, opt, device):
    num_itr = int(total_samples/batch_size)
    model.train()
    with trange(num_itr) as t:
        for i in range(num_itr):

            x, y = make_sphere(batch_size, dim, r, device)
            x.requires_grad = True

            yp = model(x)

            loss = nn.BCEWithLogitsLoss(reduction="mean")(yp, y)

            dldx = len(x) * list(grad(loss, x, create_graph=True))[0]
            reg = lambbda * torch.norm(dldx, p=2, dim=1).mean()

            loss_reg = loss + reg

            opt.zero_grad()
            loss_reg.backward()
            opt.step()

            batch_correct = ((yp > 0).bool() == y.bool()).sum().item()
            batch_acc = batch_correct / batch_size * 100

            t.set_postfix(loss=loss.item(),
                          acc="{0:.2f}%".format(batch_acc),
                          good_alpha="nan" if not logger.log_dict["alpha/good"]
                          else "{0:.2f}%".format(logger.log_dict["alpha/good"][-1][2]))
            t.update()

            if (i+1) % err_freq == 0:
                test_acc, test_loss = test_clean(dim, r, 10000, 500, model, device)
                ana_err = analytic_err(model, r)
                alpha_stats = get_alpha_stats(model, r)

                logger.add_scalar("acc/test", test_acc, i+1)
                logger.add_scalar("acc/batch", batch_acc, i+1)
                logger.add_scalar("loss/test", test_loss, i+1)
                logger.add_scalar("loss/batch", loss.item(), i+1)
                logger.add_scalar("ana_err/avg", ana_err[0], i+1)
                logger.add_scalar("ana_err/inner", ana_err[1], i+1)
                logger.add_scalar("ana_err/outer", ana_err[2], i+1)
                logger.add_scalar("alpha/good", alpha_stats[0], i+1)
                logger.add_scalar("alpha/bad_inner", alpha_stats[1], i+1)
                logger.add_scalar("alpha/bad_outer", alpha_stats[2], i+1)
                logger.add_scalar("err_in_alpha/worst", alpha_stats[3], i+1)
                logger.add_scalar("err_in_alpha/avg", alpha_stats[4], i+1)
                logger.add_scalar("reg", reg.item(), i+1)

                if np.abs(alpha_stats[0] - 100.) < 1e-5:
                    break

    log = logger.log_dict
    return log

def test_clean(dim, r, total_samples, batch_size, model, device):
    model = model.eval()
    total_loss, total_correct = 0.,0.
    
    num_itr = int(total_samples/batch_size)
    # with trange(num_itr) as t:
    for i in range(num_itr):
        x, y = make_sphere(batch_size, dim, r, device)

        yp = model(x)
        loss = nn.BCEWithLogitsLoss(reduction = "mean")(yp,y)
        # ipdb.set_trace() 
        total_correct += ((yp>0).bool() == y.bool()).sum().item()
        total_loss += loss.item() * batch_size
        # t.update()

    test_acc = total_correct / total_samples * 100
    test_loss = total_loss / total_samples
    
    return test_acc, test_loss
               
def test_adv(param, dim, r, total_samples, batch_size, model, device):
    model = model.eval()
    total_loss, total_correct = 0.,0.

    attack_param = {'eps': param["eps"], 'num_iter': param["num_iter"], 'loss_fn': nn.BCEWithLogitsLoss()}
    num_itr = int(total_samples/batch_size)
    with trange(num_itr) as t:
        for i in range(num_itr):
            
            x, y = make_sphere(batch_size, dim, r, device)
            delta = pgd_sphere(**attack_param).generate(model,x,y)
            
            yp = model(x+delta)
            loss = nn.BCEWithLogitsLoss()(yp, y)

            batch_correct = ((yp>0).bool() == y.bool()).sum().item()
            batch_acc = batch_correct / batch_size * 100
            total_loss += loss.item() * batch_size
            
            t.set_postfix(loss = loss.item(),
                          acc = "{0:.2f}%".format(batch_acc))
            t.update()
            
    test_acc = total_correct / total_samples
    test_loss = total_loss / total_samples
    return test_acc, test_loss
