import yaml
import argparse
import os

def get_args():
    
    parser = argparse.ArgumentParser()
  
    # pass slurm job id here as save directories
    parser.add_argument("--job_id", type = int, required = True)

    # specify training method:
    parser.add_argument("--method", default = "clean")
    parser.add_argument("--perfect_model", type = bool, default = False)

    # training hyper-params:
    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument("--batch_size", type = int, default = 50)
    parser.add_argument("--total_samples", type = int, default = 50e6)
    parser.add_argument("--total_itrs", type = int, default = 4e3)
    parser.add_argument("--err_freq", type = int, default = 100)
    
    # optimizater settings:
    parser.add_argument("--lr", type = float, default = 0.0001)
    parser.add_argument("--optim", default = "adam")
    parser.add_argument("--momentum", type = float, default = 0.)
    
    # sphere geometry: radius and dimension
    parser.add_argument("-r", "--radius", type = float, default = 1.3)
    parser.add_argument("--dim", type = int, default = 500)

    # specify pgd adversaries for adversarial training:
    parser.add_argument("--pgd_eps", type = float, default = 0.01)
    parser.add_argument("--pgd_itr", type = int, default = 1)
    
    # specify lambbda used for first order regularization: 
    parser.add_argument("--lambbda", type = float, default = 0.01)

    args = parser.parse_args()

    make_dir(args)

    return args

def make_dir(args):
    root_dir = "./result/"
    _dir = root_dir + str(args.job_id) + "/"

    try:
        os.makedirs(_dir)
    except os.error:
        pass

    if not os.path.exists(_dir + "/hyper_params.yaml"):
        f = open(_dir + "/hyper_params.yaml" ,"w+")
        f.close()
    with open(_dir + "/hyper_params.yaml", "a") as f:
        # f.write(str(args).replace(", ", "\n") + "\n\n")
        f.write(yaml.dump(args))
