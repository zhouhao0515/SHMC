# Copyright © Beijing University of Posts and Telecommunications,
# School of Artificial Intelligence.


import math
import time
import datetime
from typing import Iterable, Optional
import numpy as np
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma, AverageMeter
from util.utils import reduce_tensor, MetricLogger, SmoothedValue, get_detail_images, get_mask_images
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

# class bcolors:
#     HEADER = '\033[95m'
#     OKBLUE = '\033[94m'
#     OKGREEN = '\033[92m'
#     WARNING = '\033[93m'
#     FAIL = '\033[91m'
#     ENDC = '\033[0m'
#     BOLD = '\033[1m'
#     UNDERLINE = '\033[4m'
def _kl_div(p, q):
    p_log = torch.nn.functional.log_softmax(p, dim=-1)
    q_prob = torch.nn.functional.softmax(q, dim=-1)
    return torch.nn.functional.kl_div(p_log, q_prob, reduction='batchmean', log_target=False)


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False, logger=None,
                    ):
    model.train(True)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()

    # counter
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    start = time.time()
    end = time.time()

    torch.autograd.set_detect_anomaly(True)  # 会降低速度但能精确定位问题源

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 检查输入数据是否包含 NaN 或 Inf
        # if torch.isnan(samples).any() or torch.isinf(samples).any():
        #     logger.info("Input samples contain NaN or Inf values.")
        #     continue
        # if torch.isnan(targets).any() or torch.isinf(targets).any():
        #     logger.info("Input targets contain NaN or Inf values.")
        #     continue
        
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                #input_pred, attention_map = model(samples)
                input_pred, intrisics_pred, attention_maps_pred, attention_map = model(samples)
        else: # full precision
            #input_pred, attention_map = model(samples)
            input_pred, intrisics_pred, attention_maps_pred, attention_map = model(samples)
            # if any(torch.isnan(t).any() or torch.isinf(t).any() 
            #     for t in [input_pred, intrisics_pred, attention_maps_pred]):
            #     logger.info("Model outputs contain NaN/Inf, skipping batch")
            #     continue
        
        kl_loss = 0.0
        kl_loss += _kl_div(input_pred, intrisics_pred)
        # kl_loss += _kl_div(intrisics_pred, input_pred)
        kl_loss += _kl_div(intrisics_pred, attention_maps_pred)
        kl_loss += _kl_div(attention_maps_pred, input_pred)
        # kl_loss += _kl_div(input_pred, attention_maps_pred)
        kl_loss = kl_loss / 3.0
        # kl_loss = kl_loss / 2.0     

        if attention_map != []:
            with torch.no_grad():
                detail_images = get_detail_images(samples, attention_map[:, :1, :, :], theta_detail=(0.4, 0.6), padding=0.1)

            # detail-images forward
            detail_pred, _ , _ , _ = model(detail_images)

            with torch.no_grad():
                mask_images = get_mask_images(samples, attention_map[:, 1:, :, :], theta_mask=(0.2, 0.5))

            mask_pred, _ , _ , _ = model(mask_images)

            # output = (input_pred + detail_pred + mask_pred)/3.
            # loss = criterion(input_pred, targets)/3. + \
            #     criterion(detail_pred, targets)/3. + \
            #     criterion(mask_pred, targets)/3. 
            
            output = (input_pred + detail_pred + mask_pred+ intrisics_pred + attention_maps_pred)/5.   
            loss = (0.2*criterion(input_pred, targets).clamp(max=1e4) + \
                0.2*criterion(detail_pred, targets).clamp(max=1e4)  + \
                0.2*criterion(mask_pred, targets).clamp(max=1e4)  + \
                0.2*criterion(intrisics_pred, targets).clamp(max=1e4)  + \
                0.2*criterion(attention_maps_pred, targets).clamp(max=1e4)) + \
                kl_loss* 0.2  # 添加0.1的权重系数  三专家的模型
            # output = (input_pred + detail_pred + mask_pred + intrisics_pred)/4.   
            # loss = (0.25*criterion(input_pred, targets).clamp(max=1e4) + \
            #     0.25*criterion(detail_pred, targets).clamp(max=1e4)  + \
            #     0.25*criterion(mask_pred, targets).clamp(max=1e4)  + \
            #     0.25*criterion(intrisics_pred, targets).clamp(max=1e4)) + \
            #     kl_loss* 0.1  # 添加0.1的权重系数
        else:
            # loss = criterion(input_pred, targets)
            # output = input_pred
            output = (input_pred + intrisics_pred + attention_maps_pred)/3.
            loss = (criterion(input_pred, targets).clamp(max=1e4) /3. + \
                criterion(intrisics_pred, targets).clamp(max=1e4) /3. + \
                criterion(attention_maps_pred, targets).clamp(max=1e4) /3.) +\
                kl_loss* 0.2
            # output = (input_pred + intrisics_pred)/2.
            # loss = (criterion(input_pred, targets).clamp(max=1e4) /2. + \
            #     criterion(intrisics_pred, targets).clamp(max=1e4) /2.) +\
            #     kl_loss* 0.1

        loss_value = loss.item()
        # if torch.isnan(loss).any() or torch.isinf(loss).any():
        #     logger.warning(f"检测到非法损失值: {loss.item()}, 输入数据统计: mean={samples.mean().item()}, std={samples.std().item()}")
        #     optimizer.zero_grad()
        #     continue

        
        # with torch.no_grad():
        #     # 检查中间输出是否合法
        #     for name, tensor in [('input_pred', input_pred), 
        #                        ('intrisics_pred', intrisics_pred),
        #                        ('attention_maps_pred', attention_maps_pred)]:
        #         if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        #             logger.error(f"检测到非法中间输出: {name}")
        #             logger.error(f"{name}统计: mean={tensor.mean()}, std={tensor.std()}, min={tensor.min()}, max={tensor.max()}")
        #             raise RuntimeError(f"非法中间输出: {name}")
                
        if not math.isfinite(loss_value): # this could trigger if using AMP
            logger.info("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            loss.backward()

            # with torch.no_grad():
            #     for name, param in model.named_parameters():
            #         if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
            #             logger.warning(f"参数 {name} 检测到非法梯度，均值: {param.grad.mean().item()}")
            #             param.grad.data.zero_()

            # try:
            #     # 添加更安全的梯度裁剪
            #     torch.nn.utils.clip_grad_norm_(
            #         parameters=model.parameters(),
            #         max_norm=0.3,
            #         error_if_nonfinite=True  # 显式检测非法梯度
            #     )
            # except RuntimeError as e:
            #     logger.error(f"梯度裁剪失败: {str(e)}")
            #     optimizer.zero_grad()
            #     continue
            # try:
            #     loss.backward()
            #     if (data_iter_step + 1) % update_freq == 0:
            #         optimizer.step()
            #         optimizer.zero_grad()
            #         if model_ema is not None:
            #             model_ema.update(model)
            # except RuntimeError as e:
            #     logger.error(f"反向传播失败: {str(e)}")
            #     optimizer.zero_grad()
            #     continue
            # with torch.autograd.detect_anomaly():
            #     loss.backward()

            if (data_iter_step + 1) % update_freq == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    if model_ema is not None:
                        model_ema.update(model)
            

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_logger:
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': loss_value,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)
            if class_acc:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc': class_acc}, commit=False)
            if use_amp:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})

        loss_meter.update(loss.item(), targets.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if data_iter_step % update_freq == 0:
            lr = max_lr
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - data_iter_step)
            logger.info(
                f'Train: [{epoch}][{data_iter_step}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    train_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    return train_stat

@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False, logger=None, update_freq=1):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    all_preds = []
    all_targets = []

    # switch to evaluation mode
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    # statistics
    record_truth = np.array([])
    record_pred = np.array([])
    record_feature = torch.tensor([])

    end = time.time()
    idx = 0
    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, 10, header):
            images = batch[0]
            target = batch[-1]
            label_true = target.numpy().squeeze()
            record_truth = np.concatenate((record_truth, label_true))

            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            if use_amp:
                with torch.cuda.amp.autocast():
                    #input_pred, attention_map = model(images)
                    input_pred, intrisics_pred, attention_maps_pred, attention_map = model(images)
            else:
                #input_pred, attention_map = model(images)
                input_pred, intrisics_pred, attention_maps_pred, attention_map = model(images)

            if attention_map != []:
                detail_images = get_detail_images(images, attention_map, theta_detail=0.1, padding=0.05)
                detail_pred, _, _, _ = model(detail_images)
                # output = (input_pred + detail_pred)/2.
                # loss = criterion(output, target)
                output = (input_pred + detail_pred + intrisics_pred + attention_maps_pred)/4.
                #output = (intrisics_pred + attention_maps_pred)/3.
                #output = attention_maps_pred
                loss = criterion(output, target)
            else:
                # loss = criterion(input_pred, target)
                # output = input_pred
                output = (input_pred + intrisics_pred + attention_maps_pred)/3.
                #output = (intrisics_pred  + attention_maps_pred)/2.
                #output = attention_maps_pred
                loss = criterion(output, target)

            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            # record
            _, pred = torch.max(output, dim=1)
            pred = pred.cpu().numpy().squeeze()
            record_pred = np.concatenate((record_pred, pred))

            acc1 = reduce_tensor(acc1)
            acc5 = reduce_tensor(acc5)
            loss = reduce_tensor(loss)

            loss_meter.update(loss.item(), target.size(0))
            acc1_meter.update(acc1.item(), target.size(0))
            acc5_meter.update(acc5.item(), target.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % update_freq == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                if logger:
                    logger.info(
                        f'Test: [{idx}/{len(data_loader)}]\t'
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                        f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                        f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                        f'Mem {memory_used:.0f}MB')
            idx += 1

            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # 在计算完metrics后添加混淆矩阵计算和绘制
    # if logger is not None:
    #     output_dir = args.output_dir if 'args' in globals() else './'
    #     np.savetxt(os.path.join(output_dir, 'all_targets-all.txt'), all_targets, fmt='%d')
    #     np.savetxt(os.path.join(output_dir, 'all_preds-all.txt'), all_preds, fmt='%d')
    #     logger.info(f"预测结果、标签和文件名已保存到 {output_dir}")

    # 绘制混淆矩阵的函数
    # def plot_confusion_matrix_from_data(targets, preds, output_path):
    #     cm = confusion_matrix(targets, preds)
    #     cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
    #     plt.figure(figsize=(12, 10))
    #     ax = sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
    #                     xticklabels=True, yticklabels=True,
    #                     cbar_kws={'label': 'Accuracy Percentage (%)'})
        
    #     ax.set_title('Confusion Matrix with Accuracy Percentage')
    #     ax.set_xlabel('Predicted Label')
    #     ax.set_ylabel('True Label')
        
    #     plt.xticks(rotation=45)
    #     plt.yticks(rotation=0)
    #     plt.tight_layout()
    #     plt.savefig(output_path, dpi=300, bbox_inches='tight')
    #     plt.close()
        
    #     # 记录每个类别的准确率
    #     class_accuracies = cm.diagonal() / cm.sum(axis=1) * 100
    #     for i, acc in enumerate(class_accuracies):
    #         logger.info(f'类别 {i} 准确率: {acc:.2f}%')
    #     return cm_percent

    # # 绘制并保存混淆矩阵
    # if logger is not None:
    #     output_dir = args.output_dir if 'args' in globals() else './'
    #     cm_path = os.path.join(output_dir, 'confusion_matrix_percent.png')
    #     plot_confusion_matrix_from_data(all_targets, all_preds, cm_path)
    #     logger.info(f"混淆矩阵已保存到 {cm_path}")
    if logger is not None:
        load_and_plot_confusion_matrix(
        targets_path='./all_targets.txt',
        preds_path='./all_preds.txt',
        output_dir='./'
    )
    # if logger is not None:  # 只在主进程中绘制
    #     # 计算混淆矩阵
    #     cm = confusion_matrix(all_targets, all_preds)

    #     cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
    #     # 绘制混淆矩阵
    #     plt.figure(figsize=(12, 10))
    #     ax = sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
    #                     xticklabels=True, yticklabels=True,
    #                     cbar_kws={'label': 'Accuracy Percentage (%)'})
        
    #     # 设置标签和标题
    #     ax.set_title('Confusion Matrix with Accuracy Percentage')
    #     ax.set_xlabel('Predicted Label')
    #     ax.set_ylabel('True Label')
        
    #     # 调整标签旋转角度
    #     plt.xticks(rotation=45)
    #     plt.yticks(rotation=0)
        
    #     # 保存混淆矩阵图片
    #     output_dir = args.output_dir if 'args' in globals() else './'
    #     plt.tight_layout()  # 防止标签被截断
    #     plt.savefig(os.path.join(output_dir, 'confusion_matrix_percent.png'), dpi=300, bbox_inches='tight')
    #     plt.close()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # metric
    acc = accuracy_score(record_truth, record_pred)
    precision = precision_score(record_truth, record_pred, average='weighted')
    recall = recall_score(record_truth, record_pred, average='weighted')
    f1 = f1_score(record_truth, record_pred, average='weighted')
    logger.info(f'[Info] acc: {acc}, precision: {precision}, recall: {recall}, f1: {f1}')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def load_and_plot_confusion_matrix(targets_path, preds_path, output_dir='./'):
    """从文件加载预测结果和真实标签并绘制混淆矩阵"""
    all_targets = np.loadtxt(targets_path, dtype=int)
    all_preds = np.loadtxt(preds_path, dtype=int)
    
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(all_targets, all_preds)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    ax = sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=True, yticklabels=True,
                    cbar_kws={'label': 'Accuracy Percentage (%)'})
    
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    output_path = os.path.join(output_dir, 'confusion_matrix_from_file.pdf')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 打印每个类别的准确率
    class_accuracies = cm.diagonal() / cm.sum(axis=1) * 100
    for i, acc in enumerate(class_accuracies):
        print(f'类别 {i} 准确率: {acc:.2f}%')
    
    print(f"混淆矩阵已保存到 {output_path}")
    return cm_percent
