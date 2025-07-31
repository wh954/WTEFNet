# WTEFNet: Real-Time Low-Light Object Detection for Advanced Driver Assistance Systems

Hao Wu, Junzhou Chen, Ronghui Zhang, Nengchao Lyu, Hongyu Hu, Yanyong Guo, and Tony Z. Qiu

[Paper Download](https://arxiv.org/abs/2505.23201)


* The download link for the GSN dataset is below:

<table>
<tbody>
  <tr>
    <th>Google Drive</th>
    <th colspan="2"> <a href="https://drive.google.com/file/d/1i1XjyKFmXOUWc4w7X118tUlw2ucNnL2T/view?usp=sharing">Download</a> </th>
  </tr>
   <tr>
    <th>Baidu Cloud</th>
    <th colspan="2"> <a href="https://pan.baidu.com/s/1-KO3Kls4w-QFEHK2ntyknQ?pwd=uvbm">Download</a> (Extraction code: uvbm)</th> 
  </tr>
</tbody>
</table>

* The file structure of the downloaded dataset is as follows.

```
output
├── images
│   ├── train
│   ├── val
│   │── test
├── labels
│   ├── train
│   ├── val
│   │── test
```

## 👐 Hands-On Guide

### Step 1 — Requirements
To install requirements:
```
pip install -e .
```

### Step 2 — Prepare the configuration file
```
model: ./ultralytics/cfg/models/wtefnet.yaml
dataset: ./ultralytics/data/mydataset.yaml
```

### Step 3 - Train
```
cd ultralytics
python train.py
```

### Step 4 - Test
```
python test.py
```

## Citation
If you use WTEFNet or GSN dataset, please consider citing:
```
@article{wu2025wtefnet,
  title={WTEFNet: Real-Time Low-Light Object Detection for Advanced Driver-Assistance Systems},
  author={Wu, Hao and Chen, Junzhou and Zhang, Ronghui and Lyu, Nengchao and Hu, Hongyu and Guo, Yanyong and Qiu, Tony Z},
  journal={arXiv preprint arXiv:2505.23201},
  year={2025}
}
```