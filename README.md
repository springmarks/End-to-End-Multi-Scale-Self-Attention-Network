# RSIE

## Title  
**Enhancing Synthetic Speech Detection with an End-to-End Multi-Scale Self-Attention Network**

## Dataset  
The dataset used in the paper can be found here:  
- [ASVspoof 2019 LA](https://www.cnblogs.com/ZigHello/p/16139075.html)  
- [ASVspoof 2021 LA](https://www.asvspoof.org/index2021.html)  

To extract the features:  
1. Run `Feature_Engineering/CQT/cqt_extract.py` to get CQT features.  
2. Run `Feature_Engineering/LFCC/reload_data.py` to get LFCC features.

## Dependencies  
- **Hardware**: GeForce RTX 4090 (Recommended for optimal performance)  

## Train
Run experiment.py

## test
Run Result_sum_loss/test_dual.py
  

  

