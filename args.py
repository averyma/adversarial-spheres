import argparse

def get_args():
    
    parser = argparse.ArgumentParser()
    
    # specify training method:
    parser.add_argument("--method", default = "adv")
    parser.add_argument("--perfect_model", type = bool, default = False)

    # training hyper-params:
    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument("--lr", type = float, default = 0.0001)
    parser.add_argument("--batch_size", type = int, default = 50)
    parser.add_argument("--total_samples", type = int, default = 50e6)
    parser.add_argument("--total_itrs", type = int, default = 4e3)
    parser.add_argument("--err_freq", type = int, default = 100)

    # sphere geometry: radius and dimension
    parser.add_argument("-r", "--radius", type = float, default = 1.3)
    parser.add_argument("--dim", type = int, default = 500)

    # specify pgd adversaries for adversarial training:
    parser.add_argument("--pgd_alpha", type = float, default = 0.01)
    parser.add_argument("--pgd_itr", type = int, default = 1)
    # parser.add_argument()
    args = parser.parse_args()

    return args

