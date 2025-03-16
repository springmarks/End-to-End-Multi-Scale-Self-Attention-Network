

import argparse
import os
import shutil
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from data_utils import ASVspoof2019
from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf
from tqdm import tqdm
import eval_metrics as em
import numpy as np
from sklearn.metrics import confusion_matrix

def gpu_memory_info():
    allocated_memory = torch.cuda.memory_allocated() / 1024**2  # MB
    cached_memory = torch.cuda.memory_reserved() / 1024**2  # MB
    max_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
    return allocated_memory, cached_memory, max_allocated


###########################################
#计算推理时间和GPU使用情况

# import torch
# import time

# lfcc_path="anti-spoofing_lfcc_model_70.pt"
# cqt_path="anti-spoofing_cqt_model_70.pt"
# model_lfcc = torch.load(lfcc_path, map_location="cuda")
# model_cqt = torch.load(cqt_path, map_location="cuda")
# # model_lfcc = model_lfcc.to('cuda')
# # model_cqt = model_cqt('cuda')

# # 创建一个输入数据（假设是一个批次的图像）
# cqt_data = torch.randn(1, 1, 100, 750).to('cuda')
# lfcc_data = torch.randn(1, 1, 60, 750).to('cuda')


# # 记录开始时间
# start_time = time.time()

# # 进行推理
# with torch.no_grad():  # 不需要计算梯度
#     output_cqt,_=model_cqt(cqt_data)
#     output_lfcc,_ = model_lfcc(lfcc_data)
    
# # 计算推理时间
# end_time = time.time()
# inference_time = end_time - start_time

# print(f'推理时间: {inference_time:.4f}秒')
# allocated, cached, max_allocated = gpu_memory_info()
# print(f"Allocated Memory: {allocated:.2f} MB")
# print(f"Cached Memory: {cached:.2f} MB")
# print(f"Max Allocated Memory: {max_allocated:.2f} MB")
####################################################
# def compute_det_curve(target_scores, nontarget_scores):

#     n_scores = target_scores.size + nontarget_scores.size
#     all_scores = np.concatenate((target_scores, nontarget_scores))
#     labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

#     # Sort labels based on scores
#     indices = np.argsort(all_scores, kind='mergesort')
#     labels = labels[indices]

#     # Compute false rejection and false acceptance rates
#     tar_trial_sums = np.cumsum(labels)
#     nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

#     frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
#     far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
#     thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

#     return frr, far, thresholds

# def calculate_EER2021(target_scores, nontarget_scores):
#     """ Returns equal error rate (EER) and the corresponding threshold. """
#     frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
#     abs_diffs = np.abs(frr - far)
#     min_index = np.argmin(abs_diffs)
#     eer = np.mean((frr[min_index], far[min_index]))
#     return eer, thresholds[min_index]

#/root/autodl-tmp/deep_learning/End-to-End/Enter_End-to-End/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt
#/root/autodl-tmp/deep_learning/End-to-End/Enter_End-to-End/ASVspoof2021_LA_eval/trial_metadata.txt
def test_model(lfcc_path,cqt_path, device):

    model_lfcc = torch.load(lfcc_path, map_location="cuda")
    model_cqt = torch.load(cqt_path, map_location="cuda")
    model_lfcc = model_lfcc.to(device)
    model_cqt = model_cqt.to(device)

    test_set = ASVspoof2019(data_path_lfcc='/root/autodl-tmp/deep_learning/End-to-End/Enter_End-to-End/LFCCFeatures/',data_path_cqt='/root/autodl-tmp/deep_learning/End-to-End/Enter_End-to-End/CQTFeatures_new/',data_protocol='/root/autodl-tmp/deep_learning/End-to-End/Enter_End-to-End/ASVspoof2021_LA_eval/trial_metadata.txt',
                            access_type='LA',data_part='2021eval',feat_length=750,padding='repeat')
    testDataLoader = DataLoader(test_set, batch_size=5,shuffle=False, num_workers=0,collate_fn=test_set.collate_fn)
    model_lfcc.eval()
    model_cqt.eval()
    


    with open(os.path.join('/root/autodl-tmp/deep_learning/End-to-End/Enter_End-to-End/End-to-End-Dual-Branch-Network-Towards-Synthetic-Speech-Detection-main/Result_sum_loss', 'checkpoint_cm_score_new.txt'), 'w') as cm_score_file:
        for i, (lfcc,cqt,labels,type) in enumerate(tqdm(testDataLoader)):
            lfcc = lfcc.unsqueeze(1).float().to(device)
            cqt = cqt.unsqueeze(1).float().to(device)
            labels = labels.to(device)
            # print(lfcc.size())
            # print(cqt.size())

            _, score_lfcc = model_lfcc(lfcc)
            scores_lfcc = F.softmax(score_lfcc, dim=1)[:, 0]
    
            score = scores_lfcc

            _, score_cqt = model_cqt(cqt)
            scores_cqt = F.softmax(score_cqt, dim=1)[:, 0]

            score = torch.add(scores_lfcc,scores_cqt)
            score = torch.div(score,2)


            for j in range(labels.size(0)):
                cm_score_file.write(
                    'A%02d %s %s\n' % ( type[j].data,
                                          "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                          score[j].item()))
    
    eer_cm, min_tDCF = compute_eer_and_tdcf(os.path.join('/root/autodl-tmp/deep_learning/End-to-End/Enter_End-to-End/End-to-End-Dual-Branch-Network-Towards-Synthetic-Speech-Detection-main/Result_sum_loss', 'checkpoint_cm_score_new.txt'),'/root/autodl-tmp/deep_learning/End-to-End/Enter_End-to-End/')
    
    #eer_cm, _ = calculate_EER2021(score[labels==0], score[labels==1])[0]
    return eer_cm, min_tDCF

import os
import re

def test(model_dir, device):
    
    folder_path="/root/autodl-tmp/deep_learning/End-to-End/Enter_End-to-End/End-to-End-Dual-Branch-Network-Towards-Synthetic-Speech-Detection-main/Result_sum_loss" 
    data_path=os.path.join(folder_path,'dev_loss.log')
    write_path = os.path.join(folder_path , 'test_loss.log')
    
    #4号
    lfcc_path = "anti-spoofing_lfcc_model_70.pt"
    cqt_path = "anti-spoofing_cqt_model_70.pt"
    #2号
    # lfcc_path = "anti-spoofing_lfcc_model_104-Copy1.pt"
    # cqt_path = "anti-spoofing_cqt_model_104-Copy1.pt"
    # #1号
    # lfcc_path = "anti-spoofing_lfcc_model_67.pt"
    # cqt_path = "anti-spoofing_cqt_model_67.pt"

    lfcc_path = os.path.join(model_dir, lfcc_path)
    cqt_path = os.path.join(model_dir, cqt_path)
    test_model(lfcc_path,cqt_path, device)
#     with open(data_path, 'r') as file:
#         for line in file:
#             match = re.search(r'loss:([\d\.]+)\s+val_eer:([\d\.]+)', line)
#             # print(match)
#             if match:
#                 # 提取并赋值给变量 a 和 b
#                 loss = float(match.group(1))  # loss
#                 eer = float(match.group(2))  # val_eer
#                 # print(eer)
#                 if eer<0.001:
#                     word = line.split()
#                     item=int(word[0])
#                     if item>=60:
#                         print(item)
#                         lfcc_path=os.path.join(folder_path,'checkpoint_new','anti-spoofing_lfcc_model_%d.pt' % (item))
#                         cqt_path=os.path.join(folder_path,'checkpoint_new','anti-spoofing_cqt_model_%d.pt' % (item))
#                         eer, DCF = test_model(lfcc_path,cqt_path, device)

#                         with open(os.path.join(write_path),'a') as log:
#                             log.write(f"{item}\t{eer}\t{DCF}\n")
                            
                            
#89这一轮的参数还不错，但是依旧没有原来的好
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m', '--model_dir', type=str, default="/root/autodl-tmp/deep_learning/End-to-End/Enter_End-to-End/End-to-End-Dual-Branch-Network-Towards-Synthetic-Speech-Detection-main/Result_sum_loss/")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test(args.model_dir, args.device)
