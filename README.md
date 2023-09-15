# com-CNN：Domain relation extraction from noisy Chinese texts
This is the repository for our paper entitled with Domain relation extraction from noisy Chinese texts [here](https://www.sciencedirect.com/science/article/pii/S0925231220311917) .

Requirements:
----
* python=3.6
* Pytorch>0.4.1
* Numpy
* scikit-learn

Datasets:
----

* Wiki: contact with the authors of `Exploratory Neural Relation Classification for Domain Knowledge Acquisition`
* Baike: https://github.com/celtics7/BaiduBaike

Usage:
----
* prepare data <br>
python prepare_data.py

* training <br>
python train.py 

* testing <br>
python test.py

****
The data preprocessing method can be seen at https://github.com/ningpang/data_preprocess.
****
