import torch
import torch.optim as optim
from args import get_args
from model import quad
from utils_general import *
from sphere import *

def get_optim(optimizer, lr, momentum, model):
    if optimizer == "adam":
        opt = optim.Adam(model.parameters(), lr = lr)
    elif optimizer == "sgd":
        opt = optim.SGD(model.parameters(), lr = lr, momentum = momentum)
    else:
        raise NotImplementedError("The specified optimizer is not considered!")

    return opt

def train(argu, model, opt, device):
    
    if argu.method == "clean":
        stats = train_clean(argu.dim, argu.radius, 
                            argu.total_samples, argu.batch_size, argu.err_freq,
                            model, opt, device)

    elif argu.method == "truemax":
        stats = train_truemax(argu.dim, argu.radius,
                              argu.total_itrs, argu.err_freq, 
                              model, opt, device)

    elif argu.method == "adv":
        param = {"eps": argu.pgd_eps, "num_iter": argu.pgd_itr}
        stats = train_adv(param, argu.dim, argu.radius,
                          argu.total_samples, argu.batch_size, argu.err_freq,
                          model, opt, device)
    else:
        raise NotImplementedError("Training method not implemented!")

    return stats

def main():
    
    args = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed_everything(args.seed)
    model = quad().to(device)
    if args.perfect_model:
        model = make_perfect_model(model, args.radius, device)

    opt = get_optim(args.optim, args.lr, args.momentum, model)

    stats = train(args, model, opt, device)

    fig = plot_stats(stats, True)

    fig.savefig("./result/" + str(args.job_id) + "/result.png")
    torch.save(stats, "./result/" + str(args.job_id) + "/stats.pt")

if __name__ == "__main__":
    main()
