# Nopain
**NoPain: No-box Point Cloud Attack via Optimal Transport Singular Boundary**


## Introduction
This repository offers a No-box Point Cloud Attack, capable of efficiently generating transferable adversarial samples without requiring iterative updates or guidance from surrogate classifiers.

Adversarial attacks exploit the vulnerability of deep models against adversarial samples. Existing point cloud attackers are tailored to specific models, iteratively optimizing perturbations based on gradients in either a white-box or black-box setting. Despite their promising attack performance, they often struggle to produce transferable adversarial samples due to overfitting the specific parameters of surrogate models. To overcome this issue, we shift our focus to the data distribution itself and introduce a novel approach named NoPain, which employs optimal transport (OT) to identify the inherent singular boundaries of the data manifold for cross-network point cloud attacks. Specifically, we first calculate the OT mapping from noise to the target feature space, then identify singular boundaries by locating non-differentiable positions. Finally, we sample along singular boundaries to generate adversarial point clouds. Once the singular boundaries are determined, NoPain can efficiently produce adversarial samples without the need of iterative updates or guidance from the surrogate classifiers. Extensive experiments demonstrate that the proposed end-to-end method outperforms baseline approaches in terms of both transferability and efficiency, while also maintaining notable advantages even against defense strategies.

## Results


## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/cognaclee/nopain
    cd nopain
    ```
2. Download the datasets and place them in the `data/` directory:
    ```
    data/
    ├── shapenetcore_partanno_segmentation_benchmark/
    ├── ScanObjectNN/
    └── modelnet40_normal_resampled/
    ```
3. Download the pre-trained models from the [Google Drive](https://drive.google.com/drive/folders/1K0i1Q-77maDBT03fSGRQzHXA1bvgNSD5?usp=drive_link) link and place them in the `pretrained/` directory:

    ```bash
    # Create the pretrained directory if it doesn't exist
    mkdir -p pretrained
    ···
## Acknowledgments
```
* https://github.com/cuge1995/SS-attack
* https://github.com/luost26/diffusion-point-cloud
* https://github.com/stevenygd/PointFlow
```


## Citation

```

```
