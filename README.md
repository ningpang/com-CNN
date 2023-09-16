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
## Citation

If you find our work useful, please kindly cite our paper:

```
@article{DBLP:journals/ijon/PangTZZX20,
  author       = {Ning Pang and
                  Zhen Tan and
                  Xiang Zhao and
                  Weixin Zeng and
                  Weidong Xiao},
  title        = {Domain relation extraction from noisy Chinese texts},
  journal      = {Neurocomputing},
  volume       = {418},
  pages        = {21--35},
  year         = {2020},
  url          = {https://doi.org/10.1016/j.neucom.2020.07.077},
  doi          = {10.1016/j.neucom.2020.07.077},
  timestamp    = {Sun, 02 Oct 2022 15:38:32 +0200},
  biburl       = {https://dblp.org/rec/journals/ijon/PangTZZX20.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
