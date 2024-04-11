# Paper
High-performance c-VEP-BCI under minimal calibration

Yining Miao, Nanlin Shi, Changxing Huang, Yonghao Song, Xiaogang Chen, Yijun Wang, Xiaorong Gao

[Expert Systems With Applications](https://www.sciencedirect.com/science/article/pii/S0957417424005451)


# Results

In this paper, we propose two decoding methods for c-VEP-BCI with minimal calibration: Linear-Modeling method and Transfer-Learning method.

## Online experiments demo

![video1](https://github.com/Miao-yining/few-shot-c-VEP-BCI/blob/main/LinearModeling.mp4)

![video2](https://github.com/Miao-yining/few-shot-c-VEP-BCI/blob/main/TransferLearning.mp4)

## The Schematic view of the proposed methods

![image](https://github.com/Miao-yining/few-shot-c-VEP-BCI/blob/main/fig1.png)

## The online cued-spelling and free-spelling BCI performace

### 1. Online cued-spelling BCI performance

||w/Linear-Modeling||w/Transfer-Learning||
|:-----:|:-----:|:-----:|:-----:|:-----:|
|Subject|Acc(%)|ITR(bpm)|ACC(%)|ITR(bpm)|
|S1|93.50|111.15|94.50|226.75|
|S2|100.00|127.73|92.00|215.85|
|S3|91.00|105.83|100.00|255.45|
|S4|92.50|108.99|94.50|226.75|
|S5|96.50|118.03|98.00|243.59|
|S6|82.50|89.47|47.00|73.12|
|S7|88.00|99.80|79.00|166.58|
|S8|92.00|107.93|63.00|115.95|
|S9|52.50|43.52|77.50|161.45|
|S10|89.50|102.78|98.50|246.25|
|Mean±STD|87.80±12.59|101.52±21.64|84.40±16.80|193.17±58.39|


### 2. Online free-spelling BCI performance

||w/Linear-Modeling||w/Transfer-Learning||
|:-----:|:-----:|:-----:|:-----:|:-----:|
|Subject|Acc(%)|ITR(bpm)|ACC(%)|ITR(bpm)|
|S1|100.00|127.73|95.00|229.02|
|S2|100.00|127.73|100.00|255.45|
|S3|91.25|106.35|100.00|255.45|
|S4|93.75|111.70|100.00|255.45|
|S5|100.00|127.73|100.00|255.45|
|Mean±STD|97.00±3.76|120.25±9.32|99.00±2.00|250.16±10.57|

# Instructions
### About the dataset:
1. All data are restored as PICKLE format, with information of channel names, stimulus sequences and EEG responses involved. 
2. The subject numbers between offline and online experiments do NOT correspond to each other, while those within the experiments correspond. 
3. We apologize for missing some online-experiment data collected from S3 due to force majeure factors. 

### About the codes:
1. Python packages required additionally: mne, sklearn. 
2. The codes of the models proposed in the study are  provided as Python Classes in this repository, convenient for plug-and-play code writing and results reproduction. 
3. In the codes, the abbreviation 'TTCA' serves as the linear-modeling method, 'TTstCCA' serves as the transfer-learning method, and 'TTfbCCA' serves as the linear-modeling based SSVEP decoding method. 

# Citation

If our work is helpful to your research, please cite it as below.

```bibtex
@article{miao2024high,
  title={High-performance c-VEP-BCI under minimal calibration},
  author={Miao, Yining and Shi, Nanlin and Huang, Changxing and Song, Yonghao and Chen, Xiaogang and Wang, Yijun and Gao, Xiaorong},
  journal={Expert Systems with Applications},
  pages={123679},
  year={2024},
  publisher={Elsevier}
}