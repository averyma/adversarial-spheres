# Adversarial spheres

Pytorch implementation [Adversarial Spheres][paper-link].

## Requiresments
```
pytorch 1.2.0
python 3.6.0
tensorflow
tqdm
numpy 1.15.0
```

## How to run

```
# regular training
python main.py --method clean
# training with worst case loss
python main.py --method truemax
# adversarial training with PGD attacks
python main.py --method adv --pgd_alpha 0.01 --pgd_itr 100
```

### Standard training
<img src = "images/clean.png">

### Training with the exact maximizer of the inner-max optimization
<img src = "images/truemax.png">

### Adversarial training with PGD examples

## Some minor typos in paper
Here I notice a few typos in the original paper, they will not affect the 
But they do matter for reproducibility.
- In Equation 3, Analytic error rate on the inner sphere should be:
$$\mathbb{P}_{\vec{x} \sim S_{0}}\left[\sum_{i=1}^{d} \alpha_{i} x_{i}^{2}>1\right] \approx 1-\Phi\left( - \frac{\mu}{\sigma}\right)$$

- Right above Equation 3, the variance is missing a square sign: 
$$\sigma^2=2 \sum_{i=1}^{d}\left(\alpha_{i}-1\right)^{2}$$

- The Gaussian distribution in the caption of Figure F5 should be:
$$N(0, {1 \over n} )$$


## Discussions

## Others
Please cite the following paper for Adversarial spheres:

@inproceedings{46623,
title	= {Adversarial Spheres},
author	= {Justin Gilmer and Luke Metz and Fartash Faghri and Sam Schoenholz and Maithra Raghu and Martin Wattenberg and Ian Goodfellow},
year	= {2018},
URL	= {https://arxiv.org/pdf/1801.02774.pdf}
}





[paper-link]: <https://arxiv.org/abs/1801.02774>



