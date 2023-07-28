"""
训练高级VPT（Phase2）
- 在Phase 1的基础上
- 冻结第一步所有参数
- 第二步增加一个类别匹配表
- 在最后L-K层插入更多的Token
"""

from tracking_boilderplates import *

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import argparse
# from torch.optim.lr_scheduler import OneCycleLR
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from datasets import create_datasets
from utils import *
from pathlib import Path
import torch.backends.cudnn

from PromptModels.GetPromptModel import build_promptmodel
from PromptModels_pool.GetPromptModel import build_promptmodel as build_promptmodel_pool

from timm.scheduler import CosineLRScheduler
from cb_loss import AGCL
from sampler import ClassPrioritySampler, ClassAwareSampler, BalancedDatasetSampler, CBEffectNumSampler

parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, default='imbalancedcifar100_100')
# parser.add_argument('--split', type=str, default='full')
parser.add_argument('--dataset', type=str, default='CUB_BTI')
parser.add_argument('--split', type=str, default='full')

parser.add_argument('--data_path', type=str, default='data/')
# parser.add_argument('--batch_size', type=int, default=128)
# parser.add_argument('--batch_size', type=int, default=64)
# parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=20)
# parser.add_argument('--batch_size', type=int, default=16)

# parser.add_argument('--base_lr', type=float, default=0.02)
# parser.add_argument('--lr', type=float, default=0.005)
# parser.add_argument('--base_lr', type=float, default=3e-5)
# parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--base_lr', type=float, default=0.0025*2)
parser.add_argument('--lr', type=float, default=0.000625*2)


parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.01)
# parser.add_argument('--weight_decay', type=float, default=0.03)
parser.add_argument('--image_size', type=int, default=224)
# parser.add_argument('--epochs', type=int, default=90)
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--warmup_epochs', type=int, default=1)
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--scheduler', type=str, default='cosine')
# parser.add_argument('--prompt_length', type=int, default=10)
parser.add_argument('--prompt_length', type=int, default=40)
# parser.add_argument('--name', type=str, default='phase2_cifar-lt') # 决定保存模型的位置
parser.add_argument('--name', type=str, default='phase2_CUB_BTI') # 决定保存模型的位置
parser.add_argument('--base_model', type=str, default='vit_base_patch16_224_in21k')
parser.add_argument('--tau', type=float, default=1.0, help='logit adjustment factor')


def setup_seed(seed):  # setting up the random seed
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main(args):
    setup_seed(42)
    # save_path = os.path.join('./save', args.name)
    if accelerator.is_main_process:
        save_path = Path('save')/args.name
        i = 0
        while True:
            try_path = save_path/ f'exp{i}'
            if not try_path.exists():
                save_path = try_path
                ensure_path(save_path.as_posix())
                break
            i+=1
        # ensure_path(save_path)
        set_log_path(save_path)
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = accelerator.device
    
    args.lr = args.base_lr * args.batch_size / 256
    # labels = torch.ones(batch_size).long()  # long ones
    norm_params = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
    normalize = transforms.Normalize(**norm_params)
    train_transforms = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
        
                transforms.RandomResizedCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    val_transforms = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
        

            transforms.Resize((args.image_size * 8 // 7, args.image_size * 8 // 7)),
            transforms.CenterCrop((args.image_size, args.image_size)),
            transforms.ToTensor(),
            normalize,  
        ])

    train_dataset, val_dataset, num_classes = create_datasets(
        args.data_path, train_transforms, val_transforms, 
        args.dataset, args.split)
    log(f"train dataset: {len(train_dataset)} samples")
    log(f"val dataset: {len(val_dataset)} samples")


    label_num_array = None
    try:
        label_num_array = np.array(train_dataset.get_img_num_per_cls())
    except:
        train_dataset.labels = np.empty(len(train_dataset), dtype=np.int64)
        label_num_array = np.zeros(num_classes)
        for i in range(len(train_dataset)):
            label_num_array[train_dataset[i][1]] += 1
            train_dataset.labels[i] = train_dataset[i][1]
    
    
    label_freq_array = label_num_array / label_num_array.sum()
    adjustments = np.log(label_freq_array ** args.tau + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.to(device)
    criterion = AGCL(cls_num_list=list(label_num_array), m=0.1, s=20, weight=None, train_cls=False, noise_mul=0.5, gamma=4.)
    criterion_ibs = AGCL(gamma_pos=0.5, gamma_neg=8.0, cls_num_list=list(label_num_array), m=0.1, s=20, weight=None, train_cls=False, noise_mul=0.5, gamma=4.)

    train_sampler = ClassAwareSampler(train_dataset, num_samples_cls=4) 
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                            sampler=train_sampler, shuffle=False, 
                              num_workers=2, pin_memory=False, 
                              drop_last=True) # 去掉最后一个不完整的batch，dualloss才需要
    train_loader_ibs = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, 
                              num_workers=2, pin_memory=False, 
                              drop_last=True) # 去掉最后一个不完整的batch，dualloss才需要
    
    val_loader = DataLoader(val_dataset, 
                            batch_size=int(args.batch_size),
                            shuffle=False)
    

    # extractor 是 phase1 的模型，用 PromptModels文件夹下的结构加载
    extractor = build_promptmodel(num_classes=num_classes, img_size=args.image_size, 
                              base_model=args.base_model, model_idx='ViT', patch_size=16,
                            Prompt_Token_num=args.prompt_length, VPT_type="Deep")  # VPT_type = "Shallow"
    # ckpt = torch.load('save/phase1_cifar-lt/exp34/max-va.pth', 'cpu')['state_dict']
    ckpt = torch.load('save/phase1_CUB_BTI/exp29/max-va.pth', 'cpu')['state_dict']
    # ckpt = torch.load('LPT_places.pth', 'cpu')['state_dict']
    if list(ckpt.keys())[0].startswith('module'):
       ckpt_new = {}
       for key in ckpt.keys():
           ckpt_new[key[7:]] = ckpt[key]
       ckpt = ckpt_new
    extractor.load_state_dict(ckpt)
    extractor.prompt_learner.head = nn.Identity() # 把phase1的head去掉，这样就可以把token留下来。

    for param in extractor.parameters():
        param.requires_grad_(False)
    extractor.eval()
    extractor = accelerator.prepare(extractor)
    # extractor 现在只是一个函数，没有反向传播。 不过也需要编译和放到GPU上
    
     # model 是 phase2 的模型，用 PromptModels_pool文件夹下的结构加载
    model = build_promptmodel_pool(num_classes=num_classes, img_size=args.image_size, base_model=args.base_model, model_idx='ViT', patch_size=16,
                            Prompt_Token_num=args.prompt_length, VPT_type="Deep")  # VPT_type = "Shallow"
    # 用phase1的参数初始化phase2的模型。 这是为了实现前面L层共享
    model.load_state_dict(ckpt, strict=False) 

    
    
    
    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader)
    # scheduler = OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=steps_per_epoch, epochs=args.epochs, pct_start=0.2)
    scheduler = CosineLRScheduler(optimizer, warmup_lr_init=1e-6, t_initial=args.epochs, cycle_decay=0.1, warmup_t=args.warmup_epochs)
    
    
    model.Freeze() # 只把prompt_learner参数启用
    model.prompt_learner.Prompt_Tokens.requires_grad_(False) # 去除phase1的内容
    # 注意Prompt_Tokens_pool 没有锁住，现在phase2要训练
    
    # model = torch.nn.parallel.DataParallel(model)
    # model = torch.compile(model, mode='max-autotune')
    
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    
    val_loader = accelerator.prepare(val_loader)


    # preds = model(data)  # (1, class_number)
    # print('before Tuning model output：', preds)

    # check backwarding tokens
    for param in model.parameters():
        if param.requires_grad:
            print_main_process(param.shape)
    max_va = -1
    
    
    # ckpt = torch.load('phase1.pth', 'cpu')['state_dict']
    # model.load_state_dict(ckpt)
    
    for epoch in range(args.epochs):
        print('epoch:',epoch)
        aves_keys = ['tl', 'ta', 'vl', 'va']
        aves = {k: Averager() for k in aves_keys}
        iter_num = 0
        model.train()
        cnt = 0
        for imgs, targets in tqdm(train_loader, desc='train', leave=False):
            # 同时加载两个loader，一个做了CB采样，一个没有做
            imgs_ibs, targets_ibs = next(iter(train_loader_ibs)) 
            if cnt > len(train_dataset) // args.batch_size:
                break
            cnt += 1
            imgs = imgs.to(device)
            targets = targets.to(device)
            imgs_ibs, targets_ibs = imgs_ibs.to(device), targets_ibs.to(device)
            
            optimizer.zero_grad()
            outputs, reduced_sim = model(imgs)
            outputs_ibs, reduced_sim_ibs = model(imgs_ibs)
            with accelerator.autocast(): # mixed precision
                loss = criterion(outputs, targets) - 0.5 * reduced_sim + max(
                0.0, (0.5 * (args.epochs - epoch) / args.epochs)
                ) * (criterion_ibs(outputs_ibs, targets_ibs) - 0.5 * reduced_sim_ibs)

            # loss.backward()
            accelerator.backward(loss)
            
            optimizer.step()
            # scheduler.step()
            acc = compute_acc(outputs, targets)
            aves['tl'].add(loss.item())
            aves['ta'].add(acc)
            
            iter_num += 1
        report_train(aves['tl'].v, aves['ta'].v, epoch)
        scheduler.step(epoch+1)
        if epoch%4==0:
            model.eval()
            total_outputs = []
            total_labels = []
            for imgs, targets in tqdm(val_loader, desc='val', leave=False):
                imgs = imgs.to(device)
                targets = targets.to(device)
                with torch.no_grad():
                    outputs, reduced_sim = model(imgs)
                    
                    outputs, targets, reduced_sim = accelerator.gather_for_metrics(
                        (outputs, targets, reduced_sim))
                    
                    _, preds = outputs.detach().cpu().topk(1, 1, True, True)
                    preds = preds.squeeze(-1)
                    total_outputs.append(preds)
                    total_labels.append(targets.detach().cpu())
                    loss = criterion(outputs, targets)
                acc = compute_acc(outputs, targets)
                aves['vl'].add(loss.item())
                aves['va'].add(acc)
                
                

                
            total_outputs = torch.cat(total_outputs, dim=0)
            total_labels = torch.cat(total_labels, dim=0)
            # per-class evaluation
            shot_cnt_stats = {
                    'total': [0, label_num_array.max(), 0, 0, 0.],
                    'many': [100, label_num_array.max(), 0, 0, 0.],
                    'medium': [20, 100, 0, 0, 0.],
                    'few': [0, 20, 0, 0, 0.],
                }
            for l in torch.unique(total_labels):
                class_correct = torch.sum((total_outputs[total_labels == l] == total_labels[total_labels == l])).item()
                test_class_count = len(total_labels[total_labels == l])
                for stat_name in shot_cnt_stats:
                    stat_info = shot_cnt_stats[stat_name]
                    if label_num_array[l] > stat_info[0] and label_num_array[l] <= stat_info[1]:
                        stat_info[2] += class_correct
                        stat_info[3] += test_class_count
            for stat_name in shot_cnt_stats:
                shot_cnt_stats[stat_name][-1] = shot_cnt_stats[stat_name][2] / shot_cnt_stats[stat_name][3] * 100.0 if shot_cnt_stats[stat_name][3] != 0 else 0.
            per_cls_eval_str = 'epoch {}, overall: {:.5f}%, many-shot: {:.5f}%, medium-shot: {:.5f}%, few-shot: {:.5f}%'.format(epoch, shot_cnt_stats['total'][-1], shot_cnt_stats['many'][-1], shot_cnt_stats['medium'][-1], shot_cnt_stats['few'][-1])
            # log(per_cls_eval_str)
            print_main_process(per_cls_eval_str)
            log_str = 'epoch {}, train {:.4f}|{:.4f}'.format(
                    epoch, aves['tl'].v, aves['ta'].v)
            log_str += ', val {:.4f}|{:.4f}'.format(aves['vl'].v, aves['va'].v)
            # log(log_str)
            print_main_process(log_str)
            # preds = model(data)  # (1, class_number)
            print_main_process('After Tuning model output：', aves['va'].v)
            save_obj = {
                'config': vars(args),
                'state_dict': model.state_dict(),
                'val_acc': aves['va'].v,
            }
            
            report_test(aves['vl'].v, aves['va'].v, epoch)
            
            if accelerator.is_main_process:
                if epoch <= args.epochs:
                    accelerator.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))
                    # torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))
                    
                    accelerator.save(save_obj, os.path.join(
                        save_path, 'epoch-{}.pth'.format(epoch)))
                    # torch.save(save_obj, os.path.join(
                    #                     save_path, 'epoch-{}.pth'.format(epoch)))

                    if aves['va'].v > max_va:
                        max_va = aves['va'].v
                        accelerator.save(save_obj, os.path.join(save_path, 'max-va.pth'))
                        # torch.save(save_obj, os.path.join(save_path, 'max-va.pth'))
                else:
                    accelerator.save(save_obj, os.path.join(save_path, 'epoch-ex.pth'))
                    # torch.save(save_obj, os.path.join(save_path, 'epoch-ex.pth'))
                # scheduler.step(epoch)

if __name__ == "__main__":
    
    
    args = parser.parse_args()
    init_trackers(args)
    
    # accelerator.init_trackers(project_name="Long-tailed Prompt Tuning",
    #                           config=args,)
                            #   init_kwargs={LoggerType.COMETML:{
                            #       "api_key" : "IU7r6xQZEkzR7BZRy0q7juJPe",
                            #       "workspace":"2catycm"
                            #   }})
    
    main(args)