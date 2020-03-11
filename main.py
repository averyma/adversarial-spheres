import torch
import torch.optim as optim
# import argparse
from args import get_args
from model import quad
from utils_general import *
from sphere import *

def main():
    
    args = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    save_name = args.method

    seed_everything(args.seed)
    model = quad().to(device)
    if args.perfect_model:
        model = make_perfect_model(model, args.radius, device)
    opt = optim.Adam(model.parameters(), lr = args.lr)

    if args.method == "clean":
        stats = train_clean(args.dim, args.radius, 
                            args.total_samples, args.batch_size, args.err_freq,
                            model, opt, device)

    elif args.method == "truemax":
        stats = train_truemax(args.dim, args.radius,
                              args.total_itrs, args.err_freq, 
                              model, opt, device)

    elif args.method == "adv":
        param = {"alpha": args.pgd_alpha, "num_iter": args.pgd_itr}
        save_name += ("_alpha" + str(args.pgd_alpha) + "_itr" + str(args.pgd_itr))
        stats = train_adv(param, args.dim, args.radius,
                          args.total_samples, args.batch_size, args.err_freq,
                          model, opt, device)

    else:
        raise NotImplementedError("Training method not implemented!")

    fig = plot_stats(stats, True)
    fig.savefig("./result/figure/" + save_name + ".png")

    torch.save(stats, "./result/stats/" + save_name + ".pt")

if __name__ == "__main__":
    main()
