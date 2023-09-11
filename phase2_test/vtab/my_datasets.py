from typing import Callable, Literal, Optional, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import VisionDataset
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.preprocessing import LabelEncoder

this_file = Path(__file__).resolve()
this_directory = this_file.parent
project_directory = Path("/data/users/yecanming/P_CV_for_Archaeology/Chinese-Bronze-Ware").resolve()
dataset_directory = project_directory/'public/data/4.datasets/中国出土青铜器/北京天津内蒙古'

class CUB_BTI(VisionDataset):
    def __init__(self, 
                root: str=dataset_directory.as_posix(),
                image_root: str = project_directory.as_posix(),
                transforms: Optional[Callable] = None,
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                type:Literal['train', 'test', 'major_classes_train', 'major_classes_test', 'single_instance_classes']='train',
                download=False, 
                target:Literal['器名', '朝代', '出土地址']='器名'):
        # transform和target_transform是对图片和标签的变换
        # https://github.com/pytorch/vision/issues/215
        super().__init__(root, transforms, transform, target_transform)
        if image_root is None: image_root = root
        self.type = type
        # 保证数据存在
        if download: self.download()
        if not self._check_integrity(): raise RuntimeError('Dataset not found or corrupted.')
        # 读取table
        root = Path(root).resolve()
        image_root = Path(image_root).resolve()
        table_path = root/f'{type}.csv'
        self.table = pd.read_csv(table_path)
        # 从table继续读取 data 和 targets
        # self.targets = self.table[target].astype(int).values
        self.targets = LabelEncoder().fit_transform(self.table[target])
        self.data = np.array([np.array(Image.open(image_root/i)) for i in self.table['图片路径'].values])
    
    def num_classes(self):
        return len(np.unique(self.targets))
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        # 父类没有任何能力, transform 要自己调用
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    def __len__(self) -> int:
        return len(self.table)
    
    def download(self):
        raise NotImplementedError
    def _check_integrity(self):
        root, type = self.root, self.type
        root = Path(root).resolve()
        if not root.exists(): raise FileNotFoundError(f'root {root} does not exist.')
        table_path = root/f'{type}.csv'
        if not table_path.exists(): raise FileNotFoundError(f'table {table_path} does not exist.')
        return True






#%%
if __name__ == '__main__':
    train_ds = CUB_BTI(root=dataset_directory.as_posix(), 
                       image_root=project_directory.as_posix(),
                       type='train')
    test_ds = CUB_BTI(root=dataset_directory.as_posix(),
                      image_root=project_directory.as_posix(),
                       type='test')
    train_ds[0]
# %%
