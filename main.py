import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader,ConcatDataset, random_split
import pandas
import argparse
import os
import random
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from model import SurPLA_FusionVAENet, l1_reg_all
from dataset import geneHistopDataloader 
from datetime import datetime
from optim import build_optimizer
from schedule import build_scheduler
from model_ema import ModelEMA
from misc import CheckpointManager, init_logger
from utils import CrossEntropySurvLoss,NLLSurvLoss,CoxSurvLoss
from sksurv.metrics import concordance_index_censored
import logging
import time
import math
from report_feature import TextFeature
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm

from sksurv.util import Surv
from iAUC import metric_calculator
from measure import get_params,get_flops

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch in args.schedule:
        args.lr = args.lr * args.lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

def accuracy(output, target, topk=1):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = topk
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:topk].float().sum()
        return correct_k.mul_(1.0 / batch_size)

def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

    return error
      
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count      

def train(model, model_ema, trainDataloader, criterion, optimizer, text_encoder, device,scheduler,args,epoch):
    train_loss = AverageMeter()
    train_vae_loss = AverageMeter()
    train_cls_loss = AverageMeter()
    model.train()
    start = time.time() 
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_sample_haz = []
    all_sample_id = []
    all_S = []

    reg_fn = l1_reg_all
    loss_reg = 0
    for id, data in enumerate(trainDataloader):
        gene, batch_y, img, batch_diag,event_time,c, sample_id = data
        diag_token = text_encoder.token(batch_diag)

        gene = gene.to(torch.float)
        event_time = event_time.to(torch.float)
        c = c.to(torch.float)
        gene = gene.cuda()
        batch_y = batch_y.cuda()
        img = img.cuda()
        event_time = event_time.cuda()
        c = c.cuda()
        diag_token = diag_token.cuda()

        text_feat = text_encoder(diag_token)
        text_feat = text_feat.float()
        label = batch_y
        optimizer.zero_grad()
        
        hazards, S, Y_hat, vae_loss  = model(img, gene, text_feat)
        cls_loss = criterion(hazards=hazards, S=S, Y=label, c=c)
        risk_scores = -torch.sum(S, dim=1).detach().cpu().numpy()
        censorships = c.detach().cpu().numpy()
        event_times = event_time.detach().cpu().numpy()
        all_risk_scores.append(risk_scores)
        all_censorships.append(censorships)
        all_event_times.append(event_times)
        all_sample_haz.append(hazards.detach().cpu().numpy())
        all_sample_id.append(sample_id)
        all_S.append(S.detach().cpu().numpy())
        loss  = cls_loss * (1-args.vae_loss) + args.vae_loss * vae_loss['loss']
        loss = loss + loss_reg
        train_vae_loss.update(vae_loss['loss'],label.size(0))

        loss.backward()
        optimizer.step()

        train_loss.update(loss,label.size(0))
        train_cls_loss.update(cls_loss,label.size(0))
        end = time.time()
        if args.sched != "step":
            scheduler.step()
        if model_ema is not None:
            model_ema.update(model)
    all_censorships = np.concatenate(all_censorships)
    all_event_times = np.concatenate(all_event_times)
    all_risk_scores = np.concatenate(all_risk_scores)
    all_sample_haz = np.concatenate(all_sample_haz)
    all_sample_id = np.concatenate(all_sample_id)
    all_S = np.concatenate(all_S)

    all_c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    survival_train  = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    _, _,  iauc, iauc_list = metric_calculator(all_survival_months = all_event_times, survival_train = survival_train, all_risk_scores = all_risk_scores, 
                        all_censorships = all_censorships, all_event_times = all_event_times, risk_by_bin = all_S)
    logger.info('Train: {} | ' 'C-index: {:.4f} | '  'iAUC: {:.8F} |' 'Loss: {:.8f} | ' 'vae_loss: {:.8f} | ' 'cls_loss: {:.8f} | '
                'LR: {:.3e} | ' 'Time:({:.2f}s) '
                        .format(
                            epoch,
                            all_c_index,
                            iauc,
                            train_loss.avg,
                            train_vae_loss.avg,
                            train_cls_loss.avg,
                            optimizer.param_groups[0]['lr'],
                            end-start,
                            ))        
    return {'top1': all_c_index,"train_loss":train_loss.avg},all_censorships,all_event_times,all_risk_scores,all_sample_haz,all_sample_id

def test(model, testDataloader, criterion, device,optimizer, text_encoder, epoch, logsuffix, args):
    model.eval()
    test_loss = AverageMeter()
    test_vae_loss = AverageMeter()
    test_cls_loss = AverageMeter()
    err = AverageMeter()

    all_labels = []
    sur_OS = []
    OS_time = []
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_hazards = []
    all_sample_id = []
    all_S = []
    with torch.no_grad():
        for id, data in enumerate(testDataloader):
            batch_x,batch_y,batch_img, batch_diag,event_time, c, sample_id  = data
            sur_OS.append(c)
            OS_time.append(event_time)

            batch_x = batch_x.to(torch.float)
            event_time = event_time.to(torch.float)
            c = c.to(torch.float)
            batch_x = batch_x.cuda()
            label = batch_y.cuda()
            batch_img = batch_img.cuda()
            event_time = event_time.cuda()
            c = c.cuda()
            diag_token = text_encoder.token(batch_diag)
            diag_token = diag_token.cuda()
            text_feat = text_encoder(diag_token)
            text_feat = text_feat.float()
            hazards, S, Y_hat, vae_loss = model(batch_img, batch_x, text_feat)
            cls_loss = criterion(hazards=hazards, S=S, Y=label, c=c)
            risk_scores = -torch.sum(S, dim=1).detach().cpu().numpy()
            censorships = c.detach().cpu().numpy()
            event_times = event_time.detach().cpu().numpy()

            all_risk_scores.append(risk_scores)
            all_censorships.append(censorships)
            all_event_times.append(event_times)
            all_hazards.append(hazards.detach().cpu().numpy())
            all_sample_id.append(sample_id)
            all_S.append(S.detach().cpu().numpy())
            loss = args.vae_loss * vae_loss['loss'] + (1-args.vae_loss) * cls_loss
            test_vae_loss.update(vae_loss['loss'],label.size(0))

            test_loss.update(loss,label.size(0))
            test_cls_loss.update(cls_loss,label.size(0))
            all_labels.extend(batch_y.tolist())
            
    all_censorships = np.concatenate(all_censorships)
    all_event_times = np.concatenate(all_event_times)
    all_risk_scores = np.concatenate(all_risk_scores)
    all_hazards = np.concatenate(all_hazards)
    all_sample_id = np.concatenate(all_sample_id)
    all_S = np.concatenate(all_S)

    all_c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    survival_train  = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    _, _,  iauc, iauc_list = metric_calculator(all_survival_months = all_event_times, survival_train = survival_train, all_risk_scores = all_risk_scores, 
                        all_censorships = all_censorships, all_event_times = all_event_times, risk_by_bin = all_S)

    all_labels_tensor = torch.tensor(all_labels)
    unique_labels, counts = torch.unique(all_labels_tensor, return_counts=True)
    print(f"test data Unique labels: {unique_labels}")
    print(f"test data Counts: {counts}")
    logger.info('{}: {} | ' 'C-index: {:.4f} | ' 'iAUC: {:.8f} | ' 'Loss: {:.8f} | '  ' vae_loss:{:.8f} |'  ' cls_loss:{:.8f} |' 
                'LR: {:.3e} '.format(
                            logsuffix,
                            epoch,
                            all_c_index,
                            iauc,
                            test_loss.avg,
                            test_vae_loss.avg,
                            test_cls_loss.avg,
                            optimizer.param_groups[0]['lr']
                            ))
    return {"test_loss": test_loss.avg,"top1": all_c_index},all_censorships,all_event_times,all_risk_scores,all_hazards,all_sample_id

def seed_torch(seed=1029):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

from args import parse_args
def main():
    args, args_text = parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = int(args.gpu_id)
    lucky_number = 3407
    seed_torch(lucky_number)
    init_logger(args)
    # save args
    logger.info(args)

    time_str = f'log_{time.strftime("%Y%m%d_%H%M%S", time.localtime())}'
    text_encoder = TextFeature()
    trainDataloader, testDataloader,gene_size, img_feat_size = geneHistopDataloader(geneFile=args.gene_dir, selectData=None, imgDir=args.img_dir,
                                           diagnostic_reports = args.diagnostic_reports, token = text_encoder,batchSize=args.batch_size, train_ratio=args.train_ration)
    token_len = 0
    image_shape = 0
    text_shape = 0
    gene_shape = 0
    for id, data in enumerate(trainDataloader):
        gene, batch_y, img, batch_diag,event_time,c, sample_id = data
        diag_token = text_encoder.token(batch_diag)
        diag_token = diag_token.cuda()
        text_feat = text_encoder(diag_token)
        token_len = text_feat.shape[1]
        gene = gene.to(torch.float)
        gene_shape = gene.shape[1:]
        text_shape = text_feat.shape[1:]
        image_shape = img.shape[1:]
        break

    l = gene_size[0]
    hw, ch = img_feat_size
    model = SurPLA_FusionVAENet(gene_dim=l,vision_dim=ch, text_in_dim = token_len, classes=args.classes)
    logger.info(model)
    # shapes = [image_shape, gene_shape, text_shape]
    # input_shapes = [(tuple(shape),) for shape in shapes]
    # logger.info(
    #     f'Model created, params: {get_params(model) / 1e6:.3f} M, '
    #     f'FLOPs: {get_flops(model, input_shape=((image_shape), (gene_shape), (text_shape),)) / 1e9:.3f} G')

    model.cuda()
    optimizer = build_optimizer(args.opt,
                                model,
                                args.lr,
                                eps=args.opt_eps,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                filter_bias_and_bn=not args.opt_no_filter,
                                nesterov=not args.sgd_no_nesterov,
                                sort_params=args.dyrep)
    if args.model_ema:
        model_ema = ModelEMA(model, decay=args.model_ema_decay)
    else:
        model_ema = None
    ckpt_manager = CheckpointManager(model,
                                     optimizer,
                                     ema_model=model_ema,
                                     save_dir=args.best_model_path
                                     )
    loss_fn = None
    if args.task_type == 'survival':
        if args.bag_loss == 'ce_surv':
            loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
        elif args.bag_loss == 'nll_surv':
            loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
        elif args.bag_loss == 'cox_surv':
            loss_fn = CoxSurvLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    criterion = loss_fn
    cudnn.benchmark = True

    if args.sched != "step":
        steps_per_epoch = len(trainDataloader)
        warmup_steps = args.warmup_epochs * steps_per_epoch
        decay_steps = args.decay_epochs * steps_per_epoch
        total_steps = args.epochs * steps_per_epoch
        scheduler = build_scheduler(args.sched,
                                    optimizer,
                                    warmup_steps,
                                    args.warmup_lr,
                                    decay_steps,
                                    args.decay_rate,
                                    total_steps,
                                    steps_per_epoch=steps_per_epoch,
                                    decay_by_epoch=args.decay_by_epoch,
                                    min_lr=args.min_lr)
    else:
        scheduler=None

    best_acc = 0
    score_auc = []
    label_auc = []
    for epoch in range(args.epochs):
        if args.sched == "step":          
            adjust_learning_rate(optimizer, epoch, args)
        
        metrics,train_sur_os,train_sur_time,train_sur_risk,train_surv_hazards,train_sample_id = train(model, model_ema, trainDataloader, criterion, optimizer, text_encoder,device, scheduler, args, epoch)
        
        metrics, sur_os,sur_time,sur_risk,surv_hazards,sample_id = test(model, testDataloader, criterion, device,optimizer, text_encoder,epoch, "Test:", args)
        if model_ema is not None:
           test(model_ema.module, testDataloader, criterion, device, optimizer, text_encoder, epoch,"EMA:",args)
        ckpts = ckpt_manager.update(epoch, metrics)
        if best_acc <= metrics['top1']:
            best_acc = metrics['top1']
            col_name1 = []
            survival_df = pandas.DataFrame({"ID": sample_id,"OS":sur_os,"sur_time":sur_time,"risk":sur_risk})
            for i in range(0, surv_hazards.shape[1]):
                col_name1.append(f"{i}_hazard")
            surv_hazards = pandas.DataFrame(surv_hazards, columns=col_name1)
            survival_df = pandas.concat([survival_df, surv_hazards], axis=1)
            file = f'test_{time_str}_survival_risk.csv'
            survival_df.to_csv(os.path.join(args.sur_ret_dir, file))

            col_name2 = []
            survival_df_train = pandas.DataFrame({"ID":train_sample_id,"OS":train_sur_os,"sur_time":train_sur_time,"risk":train_sur_risk})
            for i in range(0, train_surv_hazards.shape[1]):
                col_name2.append(f"{i}_hazard")
            train_surv_hazards = pandas.DataFrame(train_surv_hazards, columns=col_name2)
            survival_df = pandas.concat([survival_df, train_surv_hazards], axis=1)
            file = f'train_{time_str}_survival_risk.csv'
            survival_df_train.to_csv(os.path.join(args.sur_ret_dir, file))
            logger.info("save predict risk!!!!!")  

if __name__ == '__main__':
    main()