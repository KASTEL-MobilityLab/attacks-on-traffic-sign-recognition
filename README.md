# **Adversarial Attacks on Traffic Sign Classification**

Repository of the paper [**Evaluating Adversarial Attacks on Traffic Sign Classifiers beyond Standard Baselines**](https://arxiv.org/abs/2412.09150) at IEEE ICMLA 2024. 

We decouple model architectures of LISA-CNN or GTSRB-CNN from the datasets and compare them to generic models for image classification (ResNet18, EfficientNet-B0, DenseNet-121, MobileNetv2, and ShuffleNetv2). Furthermore, we compare two attack settings, inconspicuous and visible, which are usually regarded without direct comparison. Our results show that standard baselines like LISA-CNN or GTSRB-CNN are significantly more susceptible than the generic ones. 

## **Datasets**

Download the GTSRB and LISA datasets to  the `dataset/` folder. 

We use the predefined splits:
* **LISA**: 6834 images in total, 5467 train and 1367 test (80:20 split)
* **GTSRB**: 51839 images in total, 39209 train and 12630 test (75.64:24.36 split)

We used the `pkl` files for train and test from [Zhong et al.](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhong_Shadows_Can_Be_Dangerous_Stealthy_and_Effective_Physical-World_Adversarial_Attack_CVPR_2022_paper.pdf), available [here](https://drive.google.com/file/d/1Du8egeUG6XgAVf-h9IcxRz5gZvs7_Ldq/view?usp=sharing).

## **Models**

Three architectures, deliberately developed for the traffic sign classification task:

* **CNN$_{small}$** is the original LISA-CNN as proposed by [Eykholt et al.](https://openaccess.thecvf.com/content_cvpr_2018/papers/Eykholt_Robust_Physical-World_Attacks_CVPR_2018_paper.pdf). We used the PyTorch implementation by [Zhong et al.](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhong_Shadows_Can_Be_Dangerous_Stealthy_and_Effective_Physical-World_Adversarial_Attack_CVPR_2022_paper.pdf) (code avalaible [here](https://github.com/hncszyq/ShadowAttack)) and extended it to the GTSRB data.

![](traffic_sign_classification/CNN-small_s.png)

* **CNN$_{large}$** is the original GTSRB-CNN based on the multi-scale CNN by [Sermanet et al.](https://sermanet.github.io/papers/sermanet-ijcnn-11.pdf) and a later [implementation by Yadav](https://github.com/vxy10/p2-TrafficSigns). We adapted the implementation from [Zhong et al.](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhong_Shadows_Can_Be_Dangerous_Stealthy_and_Effective_Physical-World_Adversarial_Attack_CVPR_2022_paper.pdf) and extended it to the LISA dataset.

![](traffic_sign_classification/CNN-large_s.png)

* **CNN-STN**: we used the original implementation by [Garcia et al.](https://github.com/poojahira/gtsrb-pytorch) and extended it to the LISA dataset.

![](traffic_sign_classification/transformer_s.png)

and five generic image classification models with a comparable number of parameters:
* **ResNet18**
* **EfficientNet-B0**
* **DenseNet-121**
* **MobileNetv2**
* **ShuffleNetv2** with 1.0x output

Use notebooks `01_Train__Test_GTSRB_Models.ipynb` and `01_Train__Test_GTSRB_Models.ipynb` to train and evaluate models on the corresponding datasets.

#### **Model Performance on Clean Data**

| Model | Accuracy on LISA, % | Accuracy on GTSDB, % |Number of parameters |
|---------------|--------------------------------------------|--------------------|--------------------|
| CNN$_{small}$ (LISA-CNN)|99.71|98.13 |**0.73 M** |
| CNN$_{large}$ (GTSRB-CNN) |99.78|98.91|16.54 M|
| CNN-STN |**99.85**|**99.43**| 0.85 M|
| ResNet18 |**99.85**|99.18|11.18 M|
| EfficientNet-B0 |99.71|98.36|3.50 M|
| DenseNet-121 |99.63|98.09|7.98 M|
| MobileNetv2 |99.27|96.06 |2.28 M|
| ShuffleNetv2 |99.34|98.73|5.29 M|

#### **Adversarial Attacks**

Use notebooks `03_Attack_GTSRB.ipynb` and `04_Attack_LISA.ipynb` to train and evaluate invonspicuous and visible attacks on the corresponding datasets.



# Citation

If you find this code useful for your research, please cite our paper:

```latex
@InProceedings{pavlitska2024evaluating,
  author    = {Pavlitska, Svetlana and Müller, Leopold and Zöllner, J. Marius},
  title={Evaluating Adversarial Attacks on Traffic Sign Classifiers beyond Standard Baselines},
  booktitle = { International Conference on Machine Learning and Applications (ICMLA)},
  year      = {2024}
}
```
