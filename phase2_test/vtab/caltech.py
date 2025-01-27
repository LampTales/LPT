
import os
import os.path
from random import triangular
from typing import Any, Callable, List, Optional, Union, Tuple

from PIL import Image

from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset

_TRAIN_SPLIT_PERCENT = 90

class Caltech101(VisionDataset):
    """`Caltech 101 <http://www.vision.caltech.edu/Image_Datasets/Caltech101/>`_ Dataset.
    .. warning::
        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.
    Args:
        root (string): Root directory of dataset where directory
            ``caltech101`` exists or will be saved to if download is set to True.
        target_type (string or list, optional): Type of target to use, ``category`` or
            ``annotation``. Can also be a list to output a tuple with all specified
            target types.  ``category`` represents the target class, and
            ``annotation`` is a list of points from a hand-generated outline.
            Defaults to ``category``.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
        self,
        root: str,
        split: str,
        target_type: Union[List[str], str] = "category",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        type: str = 'all'
    ) -> None:
        super().__init__(os.path.join(root, "caltech101"), transform=transform, target_transform=target_transform)
        os.makedirs(self.root, exist_ok=True)
        if isinstance(target_type, str):
            target_type = [target_type]
        self.target_type = [verify_str_arg(t, "target_type", ("category", "annotation")) for t in target_type]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        # self.categories = sorted(os.listdir(os.path.join(self.root, "101_ObjectCategories")))
        # self.categories.remove("BACKGROUND_Google")  # this is not a real class

        # For some reason, the category names in "101_ObjectCategories" and
        # "Annotations" do not always match. This is a manual map between the
        # two. Defaults to using same name, since most names are fine.
        # name_map = {
        #     "Faces": "Faces_2",
        #     "Faces_easy": "Faces_3",
        #     "Motorbikes": "Motorbikes_16",
        #     "airplanes": "Airplanes_Side_2",
        # }
        # self.annotation_categories = list(map(lambda x: name_map[x] if x in name_map else x, self.categories))

        # self.index: List[int] = []
        # self.y = []
        # for (i, c) in enumerate(self.categories):
        #     n = len(os.listdir(os.path.join(self.root, "101_ObjectCategories", c)))
        #     self.index.extend(range(1, n + 1))
        #     self.y.extend(n * [i])
        # trainval_count = len(self.index)
        self.label_file = os.path.join("vtab", "caltech_labels.txt")
        if split == 'trainval':
            self.meta_file = os.path.join("vtab", "caltech_trainval.txt")
        elif split == 'test':
            self.meta_file = os.path.join("vtab", "caltech_test.txt")
        
        with open(self.label_file, "r") as f:
            self.classes = [c.strip() for c in f.readlines()]
            self.class_to_index = {}
            for i in range(len(self.classes)):
                self.class_to_index[self.classes[i]] = i
        
        with open(self.meta_file, "r") as f:
            self._image_files = [c.strip() for c in f.readlines()]
            self._labels = []
            for c in self._image_files:
                self._labels.append(self.class_to_index[c.split("/")[0]])
        
        # if type is not 'all':
        if type!='all':
            trainval_count = len(self._labels)
            train_count = (_TRAIN_SPLIT_PERCENT * trainval_count) // 100
            if type == 'train1000':
                self._image_files = self._image_files[:800] + self._image_files[train_count:train_count+200]
                self._labels = self._labels[train_count:train_count+200]
            elif type == 'train800':
                self._image_files = self._image_files[:800]
                self._labels = self._labels[:800]
            elif type == 'val200':
                self._image_files = self._image_files[train_count:train_count+200]
                self._labels = self._labels[train_count:train_count+200]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """
        import scipy.io

        image_file = self._image_files[index]
        label = self._labels[index]

        img = Image.open(
            os.path.join(
                self.root,
                "101_ObjectCategories",
                image_file,
            )
        )

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def _check_integrity(self) -> bool:
        # can be more robust and check hash of files
        return os.path.exists(os.path.join(self.root, "101_ObjectCategories"))

    def __len__(self) -> int:
        return len(self.index)

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_and_extract_archive(
            "https://drive.google.com/file/d/137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp",
            self.root,
            filename="101_ObjectCategories.tar.gz",
            md5="b224c7392d521a49829488ab0f1120d9",
        )
        download_and_extract_archive(
            "https://drive.google.com/file/d/175kQy3UsZ0wUEHZjqkUDdNVssr7bgh_m",
            self.root,
            filename="Annotations.tar",
            md5="6f83eeb1f24d99cab4eb377263132c91",
        )

    def extra_repr(self) -> str:
        return "Target type: {target_type}".format(**self.__dict__)


class Caltech256(VisionDataset):
    """`Caltech 256 <http://www.vision.caltech.edu/Image_Datasets/Caltech256/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``caltech256`` exists or will be saved to if download is set to True.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(os.path.join(root, "caltech256"), transform=transform, target_transform=target_transform)
        os.makedirs(self.root, exist_ok=True)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self.categories = sorted(os.listdir(os.path.join(self.root, "256_ObjectCategories")))
        self.index: List[int] = []
        self.y = []
        for (i, c) in enumerate(self.categories):
            n = len(
                [
                    item
                    for item in os.listdir(os.path.join(self.root, "256_ObjectCategories", c))
                    if item.endswith(".jpg")
                ]
            )
            self.index.extend(range(1, n + 1))
            self.y.extend(n * [i])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = Image.open(
            os.path.join(
                self.root,
                "256_ObjectCategories",
                self.categories[self.y[index]],
                f"{self.y[index] + 1:03d}_{self.index[index]:04d}.jpg",
            )
        )

        target = self.y[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_integrity(self) -> bool:
        # can be more robust and check hash of files
        return os.path.exists(os.path.join(self.root, "256_ObjectCategories"))

    def __len__(self) -> int:
        return len(self.index)

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_and_extract_archive(
            "https://drive.google.com/file/d/1r6o0pSROcV1_VwT4oSjA2FBUSCWGuxLK",
            self.root,
            filename="256_ObjectCategories.tar",
            md5="67b4f42ca05d46448c6bb8ecd2220f6d",
        )