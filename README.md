# RSIE

## Citation
If you use this code, please cite:  
**《Enhancing Synthetic Speech Detection with an End-to-End Multi-Scale Self-Attention Network》**  
*The Visual Computer*, 2025.

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
Run `experiment.py`

## test
Run `Result_sum_loss/test_dual.py`

## Context
Synthetic speech generated by text-to-speech (TTS) systems, speech cloning techniques, and deep learning models poses a significant challenge for the security and authenticity of voice-based systems. Current Synthetic Speech Detection (SSD) systems often lack robustness against novel synthetic algorithms. To address this, we propose an end-to-end integrated system that leverages a BT-ResNet network for effective frequency-domain feature extraction from audio signals. By incorporating both Constant Q-transform (CQT) and Linear Frequency Cepstral Coefficient (LFCC) features, our model captures a broad spectrum of audio information. The figure followed illustrates the overall architecture of the model.
https://github.com/springmarks/End-to-End-Multi-Scale-Self-Attention-Network/blob/main/images/Fig.png
\To mitigate feature redundancy, which can degrade model accuracy, we introduce a Multi-Scale Self-Attention (EMSA) module that selectively optimizes features, ensuring only the most relevant data is used for classification. Experimental results on the ASVspoof2019LA dataset demonstrate the superiority of our system, achieving an Equal Error Rate (EER) of 0.580 and a minimum tandem detection cost function (t-DCF) of 0.0188, showcasing robust performance and state-of-the-art SSD capabilities.

  

  

