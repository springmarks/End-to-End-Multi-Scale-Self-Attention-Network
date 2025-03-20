import argparse
import json
import os
import shutil
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import ASVspoof2019
from resnet_new import setup_seed, ResNet, TypeClassifier
import torch.nn.functional as F
import eval_metrics as em
from src import resnet_models
from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf
from as21_gmmresnet2 import AS21GMMResNet2Experiment


torch.set_default_tensor_type(torch.FloatTensor)

#修改完之后的代码，加入了resnext，DRAMiT相匹配的为resnet_DRAMiT.py文件
def initParams():
    parser = argparse.ArgumentParser(description=__doc__)
    # Data folder prepare
    parser.add_argument("--access_type", type=str, default='LA')
    parser.add_argument("--data_path_cqt", type=str, default='/root/autodl-tmp/deep_learning/End-to-End/Enter_End-to-End/CQTFeatures/')
    parser.add_argument("--data_path_lfcc", type=str, default='/root/autodl-tmp/deep_learning/End-to-End/Enter_End-to-End/LFCCFeatures/')
    parser.add_argument("--data_protocol", type=str, help="protocol path",
                        default='/root/autodl-tmp/deep_learning/End-to-End/Enter_End-to-End/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt')
    parser.add_argument("--out_fold", type=str, help="output folder",default='/root/autodl-tmp/deep_learning/End-to-End/Enter_End-to-End/End-to-End-Dual-Branch-Network-Towards-Synthetic-Speech-Detection-main/dual-branch_sum_loss_gmm')
    

    # Dataset prepare
    parser.add_argument("--feat_len", type=int, help="features length", default=750)
    parser.add_argument('--padding', type=str, default='repeat')
    parser.add_argument("--enc_dim", type=int, help="encoding dimension", default=256)

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=200, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=32, help="Mini batch size for training")
    parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")

    parser.add_argument('--beta_1', type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    parser.add_argument('--num_workers', type=int, default=0, help="number of workers")
    # parser.add_argument('--seed', type=int, help="random number seed", default=688)
    parser.add_argument('--seed', type=int, help="random number seed", default=42)
    # parser.add_argument('--seed', type=int, help="random number seed", default=3407)
    parser.add_argument('--lambda_', type=float, default=0.05, help="lambda for gradient reversal layer")
    parser.add_argument('--depth', type=int, default=29, help='Model depth.')
    parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
    parser.add_argument('--base_width', type=int, default=64, help='Number of channels in each group.')
    parser.add_argument('--widen_factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
    # Acceleration
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    setup_seed(args.seed)

    if not os.path.exists(args.out_fold):
        os.makedirs(args.out_fold)
    else:
        shutil.rmtree(args.out_fold)
        os.mkdir(args.out_fold)
    if not os.path.exists(os.path.join(args.out_fold, 'checkpoint_new')):
        os.makedirs(os.path.join(args.out_fold, 'checkpoint_new'))
    else:
        shutil.rmtree(os.path.join(args.out_fold, 'checkpoint_new'))
    with open(os.path.join(args.out_fold, 'args_new.json'), 'w') as file:
        file.write(json.dumps(vars(args), sort_keys=True, separators=('\n', ':')))
    with open(os.path.join(args.out_fold, 'train_loss_new.log'), 'w') as file:
        file.write("Start recording training loss ...\n")
    with open(os.path.join(args.out_fold, 'dev_loss_new.log'), 'w') as file:
        file.write("Start recording validation loss ...\n")

    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    return args


def getFakeFeature(feature,label):
    f = []
    l = []
    for i in range(0,label.shape[0]):
        if label[i]!=20:
            l.append(label[i])
            f.append(feature[i])
    f = torch.stack(f)
    l = torch.stack(l)
    return f,l


def train(args):
    torch.set_default_tensor_type(torch.FloatTensor)
    criterion = nn.CrossEntropyLoss()

    resnet_lfcc = ResNet(3, args.enc_dim, resnet_type='18', nclasses=2).to(args.device)
    resnet_cqt = ResNet(4, args.enc_dim,resnet_type='18', nclasses=2).to(args.device)
    
    # resnet_lfcc = AS21GMMResNet2Experiment(model_type='GMMResNet2',feature_type='LFCC21NN',access_type='LA')
    # resnet_cqt = AS21GMMResNet2Experiment(model_type='GMMResNet2',feature_type='LFCC21NN',access_type='LA')
    
    resnet_lfcc = torch.load('anti-spoofing_lfcc_model_151.pt').to(args.device)
    resnet_cqt = torch.load('anti-spoofing_cqt_model_151.pt').to(args.device)

    classifier_lfcc = TypeClassifier(args.enc_dim, 6, args.lambda_, ADV=True).to(args.device)
    classifier_cqt = TypeClassifier(args.enc_dim, 6, args.lambda_, ADV=True).to(args.device)

    resnet_lfcc_optimizer = torch.optim.Adam(resnet_lfcc.parameters(),lr=args.lr, betas=(args.beta_1,args.beta_2),eps=args.eps, weight_decay=1e-4)
    resnet_cqt_optimizer = torch.optim.Adam(resnet_cqt.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=1e-4)
    
    classifier_lfcc_optimizer = torch.optim.Adam(classifier_lfcc.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2),eps=args.eps, weight_decay=1e-4)
    classifier_cqt_optimizer = torch.optim.Adam(classifier_cqt.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2),eps=args.eps, weight_decay=1e-4)

    trainset = ASVspoof2019(data_path_lfcc=args.data_path_lfcc,data_path_cqt=args.data_path_cqt,data_protocol=args.data_protocol,
                            access_type=args.access_type,data_part='train',feat_length=args.feat_len,padding=args.padding)
    validationset = ASVspoof2019(data_path_lfcc=args.data_path_lfcc,data_path_cqt=args.data_path_cqt,data_protocol='/root/autodl-tmp/deep_learning/End-to-End/Enter_End-to-End/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt',
                            access_type=args.access_type,data_part='dev',feat_length=args.feat_len,padding=args.padding)
    trainDataLoader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,collate_fn=trainset.collate_fn)
    valDataLoader = DataLoader(validationset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,collate_fn=validationset.collate_fn)


    test_set = ASVspoof2019(data_path_lfcc='/root/autodl-tmp/deep_learning/End-to-End/Enter_End-to-End/LFCCFeatures/',data_path_cqt='/root/autodl-tmp/deep_learning/End-to-End/Enter_End-to-End/CQTFeatures/',data_protocol='/root/autodl-tmp/deep_learning/End-to-End/Enter_End-to-End/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt',
                            access_type='LA',data_part='eval',feat_length=750,padding='repeat')
    testDataLoader = DataLoader(test_set, batch_size=32,shuffle=False, num_workers=0,collate_fn=test_set.collate_fn)


    for epoch_num in range(args.num_epochs):
        print('\nEpoch: %d ' % (epoch_num + 1))

        resnet_lfcc.train()
        resnet_cqt.train()
        classifier_lfcc.train()
        classifier_cqt.train()

        epoch_loss = []
        epoch_lfcc_ftcloss = []
        epoch_lfcc_fcloss = []
        epoch_cqt_ftcloss = []
        epoch_cqt_fcloss = []
        for i, (lfcc,cqt,label,fakelabel) in enumerate(tqdm(trainDataLoader)):
            # if lfcc.size(0)==32:
            if True==True:
                lfcc = lfcc.unsqueeze(1).float().to(args.device) #[32,1,60,750]
                cqt = cqt.unsqueeze(1).float().to(args.device)    #[32,1,100,750]
                # lfcc = lfcc.float().to(args.device)  # [2,1,60,750]
                # cqt = cqt.float().to(args.device)
                label = label.to(args.device)
                fakelabel = fakelabel.to(args.device)

                # get fake features and forgery type label
                feature_lfcc, out_lfcc = resnet_lfcc(lfcc)
                feature_fake_lfcc,fakelabel_lfcc = getFakeFeature(feature_lfcc,fakelabel)

                # calculate ftcloss
                typepred_lfcc = classifier_lfcc(feature_fake_lfcc)
                typeloss_lfcc = criterion(typepred_lfcc, fakelabel_lfcc)

                # optimize FTCM
                classifier_lfcc_optimizer.zero_grad()
                typeloss_lfcc.backward(retain_graph=True)
                classifier_lfcc_optimizer.step()

                # get new ftcloss
                type_pred_lfcc = classifier_lfcc(feature_fake_lfcc)
                ftcloss_lfcc = criterion(type_pred_lfcc, fakelabel_lfcc)

                # calculate fcloss
                fcloss_lfcc = criterion(out_lfcc,label)

                # cqt branch
                # get fake features and forgery type label
                feature_cqt, out_cqt = resnet_cqt(cqt)
                feature_fake_cqt, fakelabel_cqt = getFakeFeature(feature_cqt, fakelabel)

                # calculate ftcloss
                typepred_cqt = classifier_cqt(feature_fake_cqt)
                typeloss_cqt = criterion(typepred_cqt, fakelabel_cqt)

                # optimize FTCM
                classifier_cqt_optimizer.zero_grad()
                typeloss_cqt.backward(retain_graph=True)
                classifier_cqt_optimizer.step()

                # get new ftcloss
                type_pred_cqt = classifier_cqt(feature_fake_cqt)
                ftcloss_cqt = criterion(type_pred_cqt, fakelabel_cqt)

                # calculate fcloss
                fcloss_cqt = criterion(out_cqt, label)

                # LOSS
                loss = ftcloss_lfcc + fcloss_lfcc + ftcloss_cqt + fcloss_cqt
                epoch_loss.append(loss.item())

                epoch_lfcc_ftcloss.append(ftcloss_lfcc.item())
                epoch_lfcc_fcloss.append(fcloss_lfcc.item())
                epoch_cqt_ftcloss.append(ftcloss_cqt.item())
                epoch_cqt_fcloss.append(fcloss_cqt.item())

                # opyimize Feature Extraction Module and Forgery Classification Module
                resnet_lfcc_optimizer.zero_grad()
                resnet_cqt_optimizer.zero_grad()
                loss.backward()
                resnet_lfcc_optimizer.step()
                resnet_cqt_optimizer.step()


        with open(os.path.join(args.out_fold,'train_loss.log'),'a') as log:
            log.write(str(epoch_num+1) + '\t' +
                      'loss:' + str(np.nanmean(epoch_loss)) + '\t' +
                      'lfcc_fcloss:' + str(np.nanmean(epoch_lfcc_fcloss)) + '\t' +
                      'cqt_fcloss:' + str(np.nanmean(epoch_cqt_fcloss)) + '\t' +
                      'lfcc_ftcloss:' + str(np.nanmean(epoch_lfcc_ftcloss)) + '\t' +
                      'cqt_ftcloss:' + str(np.nanmean(epoch_cqt_ftcloss)) + '\t' +
                      '\n'
                      )

        resnet_lfcc.eval()
        resnet_cqt.eval()
        classifier_cqt.eval()
        classifier_lfcc.eval()

        with torch.no_grad():
            dev_loss = []
            label_list = []
            scores_list = []

            for i, (lfcc,cqt,label,_) in enumerate(tqdm(valDataLoader)):
                # if lfcc.size(0)==32:
                if True==True:
                    lfcc = lfcc.unsqueeze(1).float().to(args.device)
                    # lfcc = lfcc.float().to(args.device)
                    # cqt = cqt.float().to(args.device)
                    cqt = cqt.unsqueeze(1).float().to(args.device)
                    label = label.to(args.device)

                    _, out_lfcc = resnet_lfcc(lfcc)
                    fcloss_lfcc = criterion(out_lfcc, label)
                    score_lfcc = F.softmax(out_lfcc, dim=1)[:, 0]

                    _, out_cqt = resnet_cqt(cqt)
                    fcloss_cqt = criterion(out_cqt, label)
                    score_cqt = F.softmax(out_cqt, dim=1)[:, 0]

                    score = torch.add(score_lfcc,score_cqt)
                    score = torch.div(score,2)

                    loss = fcloss_lfcc + fcloss_cqt
                    dev_loss.append(loss.item())

                    label_list.append(label)
                    scores_list.append(score)

            scores = torch.cat(scores_list,0).data.cpu().numpy()
            labels = torch.cat(label_list,0).data.cpu().numpy()
            val_eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]

            with open(os.path.join(args.out_fold, 'dev_loss.log'), 'a') as log:
                log.write(str(epoch_num + 1) + '\t' +
                          'loss:'+ str(np.nanmean(dev_loss)) + '\t' +
                          'val_eer:' + str(val_eer) + '\t' +
                          '\n')

        torch.save(resnet_lfcc, os.path.join(args.out_fold, 'checkpoint_new','anti-spoofing_lfcc_model_%d.pt' % (epoch_num + 1)))
        torch.save(resnet_cqt, os.path.join(args.out_fold, 'checkpoint_new','anti-spoofing_cqt_model_%d.pt' % (epoch_num + 1)))

#####################################################################
        if val_eer<0.001:
            resnet_lfcc.eval()
            resnet_cqt.eval()
            with open(os.path.join('/root/autodl-tmp/deep_learning/End-to-End/Enter_End-to-End/End-to-End-Dual-Branch-Network-Towards-Synthetic-Speech-Detection-main/dual-branch_sum_loss_gmm', 'checkpoint_cm_score_new.txt'), 'w') as cm_score_file:
                for i, (lfcc,cqt,labels,type) in enumerate(tqdm(testDataLoader)):
                    # if lfcc.size(0)==32:
                    if True==True:
                        lfcc = lfcc.unsqueeze(1).float().to('cuda')
                        # lfcc = lfcc.float().to('cuda')
                        # cqt = cqt.float().to('cuda')
                        cqt = cqt.unsqueeze(1).float().to('cuda')
                        labels = labels.to('cuda')

                        _, score_lfcc = resnet_lfcc(lfcc)
                        scores_lfcc = F.softmax(score_lfcc, dim=1)[:, 0]

                        score = scores_lfcc

                        _, score_cqt = resnet_cqt(cqt)
                        scores_cqt = F.softmax(score_cqt, dim=1)[:, 0]

                        score = torch.add(scores_lfcc,scores_cqt)
                        score = torch.div(score,2)


                        for j in range(labels.size(0)):
                            cm_score_file.write(
                                'A%02d %s %s\n' % ( type[j].data,
                                                      "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                                      score[j].item()))
            eer_cm, min_tDCF = compute_eer_and_tdcf(os.path.join('/root/autodl-tmp/deep_learning/End-to-End/Enter_End-to-End/End-to-End-Dual-Branch-Network-Towards-Synthetic-Speech-Detection-main/dual-branch_sum_loss_gmm', 'checkpoint_cm_score_new.txt'),'/root/autodl-tmp/deep_learning/End-to-End/Enter_End-to-End/')
            with open('/root/autodl-tmp/deep_learning/End-to-End/Enter_End-to-End/End-to-End-Dual-Branch-Network-Towards-Synthetic-Speech-Detection-main/dual-branch_sum_loss_gmm/write_loss.log','a') as log:
                        log.write(f"{epoch_num + 1}\t{eer_cm}\t{min_tDCF}\n")
                
###############################################################################################            

    return resnet_lfcc


if __name__=='__main__':
    args = initParams()
    resnet = train(args)
    model = torch.load(os.path.join(args.out_fold,'anti-spoofing_lfcc_model.pt'))

