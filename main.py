import torch
import torch.optim as optim

from args import get_args
from model import quad
from utils_general import seed_everything, metaLogger, plot_stats
from utils_sphere import make_perfect_model
from sphere import train_clean, train_truemax, train_adv, train_reg_1st

def get_optim(optimizer, lr, momentum, model):
    if optimizer == "adam":
        opt = optim.Adam(model.parameters(), lr = lr)
    elif optimizer == "sgd":
        opt = optim.SGD(model.parameters(), lr = lr, momentum = momentum)
    else:
        raise NotImplementedError("The specified optimizer is not considered!")

    return opt

def train(logger, argu, model, opt, device):
    if argu.method == "clean":
        log = train_clean(logger, argu.dim, argu.radius,
                          argu.total_samples, argu.batch_size, argu.err_freq,
                          model, opt, device)

    elif argu.method == "truemax":
        log = train_truemax(logger, argu.dim, argu.radius,
                            argu.total_itrs, argu.err_freq,
                            model, opt, device)

    elif argu.method == "adv":
        param = {"eps": argu.pgd_eps, "num_iter": argu.pgd_itr}
        log = train_adv(logger, param, argu.dim, argu.radius,
                        argu.total_samples, argu.batch_size, argu.err_freq,
                        model, opt, device)

    elif argu.method == "reg_1st":
        log = train_reg_1st(logger, argu.lambbda, argu.dim, argu.radius,
                            argu.total_samples, argu.batch_size, argu.err_freq,
                            model, opt, device)

    else:
        raise NotImplementedError("Training method not implemented!")

    return log

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = get_args()
    log_path = args.log_dir + "/" + str(args.job_id)
    logger = metaLogger(log_path)

    seed_everything(args.seed)
    model = quad().to(device)
    if args.perfect_model:
        model = make_perfect_model(model, args.radius, device)

    opt = get_optim(args.optim, args.lr, args.momentum, model)

    log = train(logger, args, model, opt, device)
    fig = plot_stats(log, log_scale=True)

    logger.add_figure("main", fig, 0)
    logger.save_log()
    logger.close()

if __name__ == "__main__":
    main()
