"""
训练高级VPT（Phase1）
- 首先是在IN21K上学习过的ViT
- 在CIFAR100-LT上面提示微调
- 加入大量bag of tricks
 - CosineLRScheduler
 - AGCL
 - ClassAwareSampler

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

from timm.scheduler import CosineLRScheduler
from cb_loss import AGCL
from sampler import ClassPrioritySampler, ClassAwareSampler, BalancedDatasetSampler, CBEffectNumSampler

parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, default='imbalancedcifar100_100')
parser.add_argument('--dataset', type=str, default='coarse_cifar100')
# parser.add_argument('--dataset', type=str, default='CUB_BTI')
parser.add_argument('--split', type=str, default='full')
parser.add_argument('--data_path', type=str, default='data/')

# parser.add_argument('--batch_size', type=int, default=128)
# batch_size = 64
batch_size = 16
parser.add_argument('--batch_size', type=int, default=batch_size)

# parser.add_argument('--batch_size', type=int, default=32)

# parser.add_argument('--batch_size', type=int, default=16)

# parser.add_argument('--base_lr', type=float, default=0.01/128*batch_size/4)
# parser.add_argument('--lr', type=float, default=0.0025/128*batch_size/4)


# parser.add_argument('--base_lr', type=float, default=0.02)
# parser.add_argument('--lr', type=float, default=0.005)
# parser.add_argument('--base_lr', type=float, default=0.0020)
# parser.add_argument('--base_lr', type=float, default=0.00025)
# parser.add_argument('--lr', type=float, default=0.000625)

# parser.add_argument('--base_lr', type=float, default=3e-5)
# parser.add_argument('--lr', type=float, default=3e-5)

parser.add_argument('--base_lr', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=0.1)


parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.01)
# parser.add_argument('--weight_decay', type=float, default=0.03)
parser.add_argument('--image_size', type=int, default=224)
# parser.add_argument('--epochs', type=int, default=90)
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--warmup_epochs', type=int, default=1)
# parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--scheduler', type=str, default='cosine')

# enable_scheduler = True
enable_scheduler = False
# parser.add_argument('--prompt_length', type=int, default=10)
# parser.add_argument('--prompt_length', type=int, default=20)
parser.add_argument('--prompt_length', type=int, default=40)
# 0.8657 acc: 0.5521

# parser.add_argument('--prompt_length', type=int, default=5)
# parser.add_argument('--name', type=str, default='phase1_cifar-lt') # 决定保存模型的位置
# parser.add_argument('--name', type=str, default='phase1_cifar100_pt100') # 决定保存模型的位置
parser.add_argument('--name', type=str, default='phase1_cifar100_coarse') # 决定保存模型的位置
# parser.add_argument('--name', type=str, default='phase1_CUB_BTI') # 决定保存模型的位置
parser.add_argument('--base_model', type=str, default='vit_base_patch16_224_in21k')
# parser.add_argument('--base_model', type=str, default='vit_large_patch16_224_in21k')
# parser.add_argument('--base_model', type=str, default='vit_base_patch16_224_miil_in21k')
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
    
    # 从 imagenet 统计出来的规律
    norm_params = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
    normalize = transforms.Normalize(**norm_params)
    train_transforms = transforms.Compose([
        # 没加之前，是 0.8769/0.5208
        # 转为灰度图  0.9142/0.5208
                # transforms.Grayscale(num_output_channels=3),
                transforms.RandomResizedCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    val_transforms = transforms.Compose([
        # 转为灰度图
            # transforms.Grayscale(num_output_channels=3), 
            
            transforms.Resize((args.image_size * 8 // 7, args.image_size * 8 // 7)),
            transforms.CenterCrop((args.image_size, args.image_size)),
            transforms.ToTensor(),
            normalize,  
        ])

    train_dataset, val_dataset, num_classes = create_datasets(
        args.data_path, train_transforms, val_transforms, 
        args.dataset, args.split)
    print_main_process(f"train dataset: {len(train_dataset)} samples")
    print_main_process(f"val dataset: {len(val_dataset)} samples")


    # TODO： 把dataset换掉，其实就是pytorch的dataset对象而已，有[]和len方法
    # 换成官网的CIFAR100，包括coarse labels和fine labels
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
    # criterion = nn.CrossEntropyLoss()

    train_sampler = ClassAwareSampler(train_dataset, num_samples_cls=4) 
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                            sampler=train_sampler, shuffle=False, 
                              num_workers=8, pin_memory=True, 
                              drop_last=False) # 去掉最后一个不完整的batch，dualloss才需要
    val_loader = DataLoader(val_dataset, 
                            batch_size=int(args.batch_size),
                            shuffle=False)
    


    model = build_promptmodel(num_classes=num_classes, img_size=args.image_size, 
                              base_model=args.base_model, model_idx='ViT', patch_size=16,
                            Prompt_Token_num=args.prompt_length, VPT_type="Deep")  # VPT_type = "Shallow"
    # test for updating 不影响正常运行
    # prompt_state_dict = model.obtain_prompt()
    # model.load_prompt(prompt_state_dict)
    # model = model.to(device)
    
    
    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader)
    # scheduler = OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=steps_per_epoch, epochs=args.epochs, pct_start=0.2)
    scheduler = CosineLRScheduler(optimizer, warmup_lr_init=args.lr, t_initial=args.epochs, cycle_decay=0.1, warmup_t=args.warmup_epochs)
    
    
    model.Freeze()
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
        print_main_process('epoch:',epoch)
        aves_keys = ['tl', 'ta', 'vl', 'va']
        aves = {k: Averager() for k in aves_keys}
        iter_num = 0
        model.train()
        for imgs, targets in tqdm(train_loader, desc='train', leave=False):
            imgs = imgs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            with accelerator.autocast(): # mixed precision
                loss = criterion(outputs, targets)
            # loss.backward()
            accelerator.backward(loss)
            
            optimizer.step()
            if enable_scheduler:
                scheduler.step()
            acc = compute_acc(outputs, targets)
            aves['tl'].add(loss.item())
            aves['ta'].add(acc)
            
            iter_num += 1
        report_train(aves['tl'].v, aves['ta'].v, epoch)

        scheduler.step(epoch+1)
        if epoch%4==0:
            model.eval()
            for imgs, targets in tqdm(val_loader, desc='val', leave=False):
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = model(imgs)
                
                outputs, targets = accelerator.gather_for_metrics((outputs, targets))
                
                loss = criterion(outputs, targets)
                acc = compute_acc(outputs, targets)
                aves['vl'].add(loss.item())
                aves['va'].add(acc)
                
                

                
            # log_str = 'epoch {}, lr: {:.4f}, train loss: {:.4f}|acc: {:.4f}'.format(
                    # epoch, scheduler.get_last_lr()[0], aves['tl'].v, aves['ta'].v)
            log_str = 'epoch {}, train {:.4f}|{:.4f}'.format(
                epoch, aves['tl'].v, aves['ta'].v)
            log_str += ', val loss: {:.4f}|acc: {:.4f}'.format(aves['vl'].v, aves['va'].v)
            # log(log_str)
            report_test(aves['vl'].v, aves['va'].v, epoch)
            print_main_process(log_str)
            
            # preds = model(data)  # (1, class_number)
            print_main_process('After Tuning model output: ', aves['va'].v)
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            
            save_obj = {
                'config': vars(args),
                'state_dict': unwrapped_model.state_dict(),
                'val_acc': aves['va'].v,
            }
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