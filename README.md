# IGB4MTL

Official implementation of _"Improvable Gap Balancing for Multi-Task Learning"_.

## Setup environment

```bash
conda create -n igb4mtl python=3.8.13
conda activate igb4mtl
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
```

Install the repo:

```bash
git clone https://github.com/YanqiDai/IGB4MTL.git
cd IGB4MTL
pip install -r requirement.txt
```

## Run experiment

Follow instruction on the experiment README file for more information regarding, e.g., datasets.

We support our IGB methods and other existing MTL methods with a unified API. To run experiments:

```bash
cd experiments/<expirimnet name>
python trainer.py --loss_method=<loss balancing method> --gradient_method=<gradient balancing method>
```
  
Here,
- `<experiment name>` is one of `[quantum_chemistry, nyuv2]`.
- `<loss balancing method>` is one of `igbv1`, `igbv2` and the following loss balancing MTL methods.
- `<gradient balancing method>` is one of the following gradient balancing MTL methods.
- Both `<loss balancing method>` and `<gradient balancing method>` are optional:
  - only using `<loss balancing method>` is to run a loss balancing method;
  - only using `<gradient balancing method>` is to run a gradient balancing method;
  - using neither is to run Equal Weighting (EW) method.
  - using both is to run a combined MTL method by both loss balancing and gradient balancing.

## MTL methods

We support the following loss balancing and gradient balancing methods.

|   Loss Balancing Method (code name)   |                                                          Paper (notes)                                                           |
|:-------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------:|
|        Equal Weighting (`ls`)         |                                                     - (linear scalarization)                                                     |
|     Random Loss Weighting (`rlw`)     |                  [A Closer Look at Loss Weighting in Multi-Task Learning](https://arxiv.org/pdf/2111.10603.pdf)                  |
|    Dynamic Weight Average (`dwa`)     |                        [End-to-End Multi-Task Learning with Attention](https://arxiv.org/abs/1803.10704)                         |
|     Uncertainty Weighting (`uw`)      | [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/pdf/1705.07115v3.pdf) |
| Improvable Gap Balancing v1 (`igbv1`) |                                                     - (our first IGB method)                                                     |
| Improvable Gap Balancing v1 (`igbv1`) |                                                    - (our second IGB method)                                                     |


| Gradient Balancing Method (code name) |                                          Paper (notes)                                           |
|:-------------------------------------:|:------------------------------------------------------------------------------------------------:|
|             MGDA (`mgda`)             |     [Multi-Task Learning as Multi-Objective Optimization](https://arxiv.org/abs/1810.04650)      |
|           PCGrad (`pcgrad`)           |           [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782)           |
|           CAGrad (`cagrad`)           | [Conflict-Averse Gradient Descent for Multi-task Learning](https://arxiv.org/pdf/2110.14048.pdf) |
|            IMTL-G (`imtl`)            |       [Towards Impartial Multi-task Learning](https://openreview.net/forum?id=IMPnRXEWpvr)       |
|         Nash-MTL (`nashmtl`)          |        [Multi-Task Learning as a Bargaining Game](https://arxiv.org/pdf/2202.01017v1.pdf)        |
