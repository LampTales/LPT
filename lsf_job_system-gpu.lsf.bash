#!/bin/sh
#BSUB -J "test" ## 任务名
## 队列名。
##BSUB -q ssc-gpu ## 6个A100 GPU
#BSUB -q 4a100-40 ## 6个A100 GPU

## BSUB -W 12:00 ## 任务最长运行时间，单位是hh:mm. 12小时太长了，提交失败
##BSUB -R "rusage[mem=10000]" ## 任务最大内存，单位是MB
##BSUB -R "select[hname=='b05u17g' || hname=='b05u08g']"

#BSUB -gpu "num=2:mode=exclusive_process"  # 申请 2 个独占模式的GPU
#BSUB -R "span[hosts=1]"  # 只有一个主机


#BSUB -e out/%J.err ## 任务的stderr文件。 %J 代表任务名
#BSUB -o out/%J.out ## 任务的stdout文件。 %J 代表任务名

module load mpi/latest
module load cuda/11.8
export I_MPI_DEBUG=5
cd $LS_SUBCWD
date
nvidia-smi

conda run -n cv accelerate launch phase2_test/train_phase1.py
date

## 运行命令
## bsub<lsf_job_system.lsf