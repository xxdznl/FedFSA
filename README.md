<<<<<<< HEAD
<p align="center" width="100%">
</p>

<div id="top" align="center">

Flexible Sharpness-Aware Personalized Federated Learning (AAAI 2025)
-----------------------------
- <a href="https://doi.org/10.1609/aaai.v39i20.35475"> Paper </a> 
- [Poster PDF](./FedFSAPoster.pdf)

<!-- **Authors:** -->

_**Xinda Xing*<sup>1</sup>, Qiugang Zhan*<sup>2 </sup>, Xiurui Xie†<sup>1</sup>, Yuning Yang<sup>1</sup>, Qiang Wang<sup>3</sup>, Guisong Liu†<sup>2</sup>**_


<!-- **Affiliations:** -->


_<sup>1</sup> University of Electronic Science and Technology of China,
<sup>2</sup> Southwest University of Finance and Economics,
<sup>3</sup> Sun Yat-sen University._
</div>

## Contents

- [Overview](#overview)
- [Baselines](#baselines)
- [Quick Start](#Quick Start)
- [Citation](#citation)
- [Acknowledgements](#acknowledgments)

## Overview
Personalized federated learning (PFL) is a new paradigm to address the statistical heterogeneity problem in federated learning. Most existing PFL methods focus on leveraging global and local information such as model interpolation or parameter decoupling. However, these methods often overlook the generalization potential during local client learning. From a local optimization perspective, we propose a simple and general PFL method, Federated learning with Flexible Sharpness-Aware Minimization (FedFSA). Specifically, we emphasize the importance of applying a larger perturbation to critical layers of the local model when using the Sharpness-Aware Minimization (SAM) optimizer. Then, we design a metric, perturbation sensitivity, to estimate the layer-wise sharpness of each local model. Based on this metric, FedFSA can flexibly select the layers with the highest sharpness to employ larger perturbation. The results show that FedFSA outperforms seven baselines by up to 8.26% in test accuracy.

## Baselines
**FedAvg**:[Communication-Efficient Learning of Deep Networks from Decentralized Data](https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf)

**FedCR**:[Communication-Efficient Learning of Deep Networks from Decentralized Data](https://proceedings.mlr.press/v202/zhang23w/zhang23w.pdf)

**FedALA**:[FedALA: Adaptive Local Aggregation for Personalized Federated Learning](https://doi.org/10.1609/aaai.v37i9.26330f)

**FedSAM/MoFedSAM**:[Generalized Federated Learning via Sharpness Aware Minimization](https://proceedings.mlr.press/v162/qu22a/qu22a.pdf)

**FedSpeed**:[FedSpeed: Larger Local Interval, Less Communication Round, and Higher Generalization Accuracy](https://openreview.net/pdf?id=bZjxxYURKT)

**FedSMOO**:[Dynamic Regularized Sharpness Aware Minimization in Federated Learning: Approaching Global Consistency and Smooth Landscape](https://proceedings.mlr.press/v202/sun23h/sun23h.pdf)

## Quick Start
Refer to the [`./FedFSA/run.sh`](./FedFSA/run.sh) script for basic usage.
For detailed hyperparameter configurations, please refer to our paper and [Appendix](./Appendix.pdf).
## Citation
```
@inproceedings{xing2025fedfsa,
    title={Flexible Sharpness-Aware Personalized Federated Learning},
    author={Xing, Xinda and Zhan, Qiugang and Xie, Xiurui and Yang, Yuning and Wang, Qiang and Liu, Guisong},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    volume={39},
    number={20},
    pages={21707--21715},
    year={2025},
    address = {Philadelphia, Pennsylvania, USA}
}
```

## Acknowledgments

This repo benefits from [FedSpeed,FedSMOO](https://github.com/woodenchild95/FL-Simulator/tree/main) and [FedCR](https://github.com/haozzh/FedCR). Thanks for their wonderful works!

=======
<p align="center" width="100%">
</p>

<div id="top" align="center">

Flexible Sharpness-Aware Personalized Federated Learning (AAAI 2025)
-----------------------------
- <a href="https://doi.org/10.1609/aaai.v39i20.35475"> Paper </a> 
- [Poster PDF](./FedFSAPoster.pdf)

<!-- **Authors:** -->

_**Xinda Xing*<sup>1</sup>, Qiugang Zhan*<sup>2 </sup>, Xiurui Xie†<sup>1</sup>, Yuning Yang<sup>1</sup>, Qiang Wang<sup>3</sup>, Guisong Liu†<sup>2</sup>**_


<!-- **Affiliations:** -->


_<sup>1</sup> University of Electronic Science and Technology of China,
<sup>2</sup> Southwest University of Finance and Economics,
<sup>3</sup> Sun Yat-sen University._
</div>

## Contents

- [Overview](#overview)
- [Baselines](#baselines)
- [Citation](#citation)
- [Acknowledgements](#acknowledgments)

## Overview
Personalized federated learning (PFL) is a new paradigm to address the statistical heterogeneity problem in federated learning. Most existing PFL methods focus on leveraging global and local information such as model interpolation or parameter decoupling. However, these methods often overlook the generalization potential during local client learning. From a local optimization perspective, we propose a simple and general PFL method, Federated learning with Flexible Sharpness-Aware Minimization (FedFSA). Specifically, we emphasize the importance of applying a larger perturbation to critical layers of the local model when using the Sharpness-Aware Minimization (SAM) optimizer. Then, we design a metric, perturbation sensitivity, to estimate the layer-wise sharpness of each local model. Based on this metric, FedFSA can flexibly select the layers with the highest sharpness to employ larger perturbation. The results show that FedFSA outperforms seven baselines by up to 8.26% in test accuracy.

## Baselines
**FedAvg**:[Communication-Efficient Learning of Deep Networks from Decentralized Data](https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf)

**FedCR**:[Communication-Efficient Learning of Deep Networks from Decentralized Data](https://proceedings.mlr.press/v202/zhang23w/zhang23w.pdf)

**FedALA**:[FedALA: Adaptive Local Aggregation for Personalized Federated Learning](https://doi.org/10.1609/aaai.v37i9.26330f)

**FedSAM/MoFedSAM**:[Generalized Federated Learning via Sharpness Aware Minimization](https://proceedings.mlr.press/v162/qu22a/qu22a.pdf)

**FedSpeed**:[FedSpeed: Larger Local Interval, Less Communication Round, and Higher Generalization Accuracy](https://openreview.net/pdf?id=bZjxxYURKT)

**FedSMOO**:[Dynamic Regularized Sharpness Aware Minimization in Federated Learning: Approaching Global Consistency and Smooth Landscape](https://proceedings.mlr.press/v202/sun23h/sun23h.pdf)

## Quick Start
Refer to the [`./FedFSA/run.sh`](./FedFSA/run.sh) script for basic usage.
For detailed hyperparameter configurations, please refer to our paper and [Appendix](./Appendix.pdf).
## Citation
```
@inproceedings{xing2025fedfsa,
    title={Flexible Sharpness-Aware Personalized Federated Learning},
    author={Xing, Xinda and Zhan, Qiugang and Xie, Xiurui and Yang, Yuning and Wang, Qiang and Liu, Guisong},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    volume={39},
    number={20},
    pages={21707--21715},
    year={2025},
    address = {Philadelphia, Pennsylvania, USA}
}
```

## Acknowledgments

This repo benefits from [FedSpeed,FedSMOO](https://github.com/woodenchild95/FL-Simulator/tree/main) and [FedCR](https://github.com/haozzh/FedCR). Thanks for their wonderful works!

>>>>>>> e9cc7ac5d8a2d7e60356d9ce7fd19d982599a635
