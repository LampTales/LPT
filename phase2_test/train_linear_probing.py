"""
实验2
训练Linear Probing，看看Probing的效果
- 首先是在IN21K上学习过的ViT
- 在CIFAR100-LT上面 仅仅进行分类学习
"""
from comet_ml import Experiment
from clearml import Task
import wandb
from tensorboardX import SummaryWriter

from accelerate import Accelerator
from accelerate.utils import LoggerType

accelerator = Accelerator(split_batches=True,
                        #   log_with=[LoggerType.WANDB, LoggerType.TENSORBOARD
                        #             # , LoggerType.COMETML
                        #             ],
                        #   logging_dir="./tensorboard"
                          ) # batch_size 始终由用户控制，不随GPU数量变化


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from PromptModels.GetPromptModel import build_promptmodel
import argparse
from torch.optim.lr_scheduler import OneCycleLR
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

from timm.scheduler import CosineLRScheduler

from datasets import create_datasets
from utils import *
from pathlib import Path

import torch.backends.cudnn

tensorboard = None
def init_trackers(args, project_name='Long-tailed Prompt Tuning', task_name='Tune VPT on CIFAR-LT'):
    if not accelerator.is_main_process: return
    # experiment = Experiment(
    #     api_key = "IU7r6xQZEkzR7BZRy0q7juJPe",
    #     project_name = project_name,
    #     workspace="2catycm"
    # )
    global tensorboard
    tensorboard = SummaryWriter('./tensorboard_log')
    # wandb.init(
    #         project=project_name,
    #         name=task_name
    # )
    # wandb.config.update(args)
    task = Task.init(project_name=project_name, 
                     task_name=task_name)
    


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='imbalancedcifar100_100')
parser.add_argument('--split', type=str, default='full')
parser.add_argument('--data_path', type=str, default='data/')
# parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=64)
# parser.add_argument('--batch_size', type=int, default=32)

parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--momentum', type=float, default=0.9)
# parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.03)
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--warmup_epochs', type=int, default=10)
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--scheduler', type=str, default='cosine')
parser.add_argument('--prompt_length', type=int, default=10)
parser.add_argument('--name', type=str, default='vit_linear_probe_cifar-lt') # 决定保存模型的位置
parser.add_argument('--base_model', type=str, default='vit_base_patch16_224_in21k')

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
    norm_params = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
    normalize = transforms.Normalize(**norm_params)
    train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    val_transforms = transforms.Compose([
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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=int(args.batch_size), shuffle=False)

    # model = build_promptmodel(num_classes=num_classes, img_size=args.image_size, 
    #                           base_model=args.base_model, model_idx='ViT', patch_size=16,
    #                         Prompt_Token_num=args.prompt_length, VPT_type="Deep")  # VPT_type = "Shallow"
    import timm

    model = timm.create_model(args.base_model,
                                    pretrained=True)
    for param in model.parameters():
        param.requires_grad_(False)
    model.head = nn.Linear(768, num_classes) # 应该默认时kaiming
    for param in model.head.parameters():
        param.requires_grad_(True)
    

    # test for updating 不影响正常运行
    # prompt_state_dict = model.obtain_prompt()
    # model.load_prompt(prompt_state_dict)
    # model = model.to(device)
    
    

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=steps_per_epoch, epochs=args.epochs, pct_start=0.2)
    criterion = nn.CrossEntropyLoss()
    
    
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
            print(param.shape)
    max_va = -1
    
    
    # ckpt = torch.load('phase1.pth', 'cpu')['state_dict']
    # model.load_state_dict(ckpt)
    
    for epoch in range(args.epochs):
        print('epoch:',epoch)
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
            scheduler.step()
            acc = compute_acc(outputs, targets)
            aves['tl'].add(loss.item())
            aves['ta'].add(acc)
            
            # accelerator.log({'train_loss':loss.item(), "train_accuracy":acc})
            if accelerator.is_main_process:
                # wandb.log({'train_loss':loss.item(), "train_accuracy":acc})
                tensorboard.add_scalar('train_loss', loss.item())
                tensorboard.add_scalar('train_accuracy', acc)
            iter_num += 1
        # print()
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
                
                # accelerator.log({'val_loss':loss.item(), "val_accuracy":acc})
                if accelerator.is_main_process:
                    # wandb.log({'val_loss':loss.item(), "val_accuracy":acc})
                    tensorboard.add_scalar('val_loss', loss.item())
                    tensorboard.add_scalar('val_accuracy', acc)
                
            log_str = 'epoch {}, lr: {:.4f}, train loss: {:.4f}|acc: {:.4f}'.format(
                    epoch, scheduler.get_last_lr()[0], aves['tl'].v, aves['ta'].v)
            log_str += ', val loss: {:.4f}|acc: {:.4f}'.format(aves['vl'].v, aves['va'].v)
            log(log_str)
            # preds = model(data)  # (1, class_number)
            print('After Tuning model output: ', aves['va'].v)
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
    accelerator.end_training()