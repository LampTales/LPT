from comet_ml import Experiment
from clearml import Task
import wandb
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from accelerate import Accelerator
from accelerate.utils import LoggerType

accelerator = Accelerator(split_batches=True,
                        #   log_with=[LoggerType.WANDB, LoggerType.TENSORBOARD
                        #             # , LoggerType.COMETML
                        #             ],
                        #   logging_dir="./tensorboard"
                          ) # batch_size 始终由用户控制，不随GPU数量变化

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
    
    
def print_main_process(*args, **kwargs):
    if accelerator.is_main_process:
        print(*args, **kwargs)

def report_train(loss, acc, epoch, iter_num, train_loader):
    if accelerator.is_main_process:
        # wandb.log({'train_loss':loss.item(), "train_accuracy":acc})
        # tensorboard.add_scalars({'train_loss':loss.item(), "train_accuracy":acc}, 
        #                         global_step=global_step)
        global_step=epoch*len(train_loader)+iter_num
        tensorboard.add_scalar('train_loss', loss.item(), global_step=global_step)
        tensorboard.add_scalar('train_accuracy', acc,  global_step=global_step)
        
def report_test(loss, acc, epoch):
    if accelerator.is_main_process:
        # wandb.log({'val_loss':loss.item(), "val_accuracy":acc})
        tensorboard.add_scalar('val_loss', loss.item(), global_step=epoch)
        tensorboard.add_scalar('val_accuracy', acc, global_step=epoch)