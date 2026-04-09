from __future__ import absolute_import
import os.path as osp
from torch.utils.data import Dataset
from PIL import Image


def _resolve_path(root, fname):
    if root is None:
        return fname
    return osp.join(root, fname)


def _load_rgb(root, fname):
    return Image.open(_resolve_path(root, fname)).convert('RGB')


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        img = _load_rgb(self.root, fname)

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, camid, index

class Preprocessor_ir(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor_ir, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, cpid, camid = self.dataset[index]
        img = _load_rgb(self.root, fname)

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, cpid, camid, index

class Preprocessor_color(Dataset):
    def __init__(self, dataset, root=None, transform=None,transform1=None):
        super(Preprocessor_color, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.transform1 = transform1
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, cpid, camid = self.dataset[index]
        img_ori = _load_rgb(self.root, fname)

        if self.transform is not None:
            img = self.transform(img_ori)
            img1 = self.transform1(img_ori)
        return img, img1,fname, pid, cpid, camid, index


class Preprocessor_inter(Dataset):
    def __init__(self, dataset_rgb, dataset_ir, root_rgb=None, root_ir=None, 
                 transform_rgb=None,transform1_rgb=None, transform_ir=None,transform1_ir=None,
                 index_rgb = None, index_ir = None):
        super(Preprocessor_inter, self).__init__()
        self.dataset_rgb = dataset_rgb
        self.root_rgb = root_rgb
        self.transform_rgb = transform_rgb
        self.transform1_rgb = transform1_rgb

        self.dataset_ir = dataset_ir
        self.root_ir = root_ir
        self.transform_ir = transform_ir
        self.transform1_ir = transform1_ir

        self.inedex_rgb = index_rgb
        self.inedex_ir = index_ir

    def __len__(self):
        if self.inedex_rgb is not None and self.inedex_ir is not None:
            return min(len(self.inedex_rgb), len(self.inedex_ir))
        return min(len(self.dataset_rgb), len(self.dataset_ir))

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname_rgb, pid_rgb, cpid_rgb, camid_rgb = self.dataset_rgb[self.inedex_rgb[index]]
        img_ori1 = _load_rgb(self.root_rgb, fname_rgb)

        if self.transform_rgb is not None:
            img10 = self.transform_rgb(img_ori1)
            img11 = self.transform1_rgb(img_ori1)

        fname_ir, pid_ir, cpid_ir, camid_ir = self.dataset_ir[self.inedex_ir[index]]
        img_ori2 = _load_rgb(self.root_ir, fname_ir)

        if self.transform_ir is not None:
            img20 = self.transform_ir(img_ori2)
            img21 = self.transform1_ir(img_ori2)
        return img10, img11, fname_rgb, pid_rgb, cpid_rgb, camid_rgb, self.inedex_rgb[index], \
            img20, img21, fname_ir, pid_ir, cpid_ir, camid_ir, self.inedex_ir[index]


class Preprocessor_inter_woaug(Dataset):
    def __init__(self, dataset_rgb, dataset_ir, root_rgb=None, root_ir=None, 
                 transform_rgb=None, transform_ir=None,
                 index_rgb = None, index_ir = None):
        super(Preprocessor_inter_woaug, self).__init__()
        self.dataset_rgb = dataset_rgb
        self.root_rgb = root_rgb
        self.transform_rgb = transform_rgb

        self.dataset_ir = dataset_ir
        self.root_ir = root_ir
        self.transform_ir = transform_ir

        self.inedex_rgb = index_rgb
        self.inedex_ir = index_ir

    def __len__(self):
        if self.inedex_rgb is not None and self.inedex_ir is not None:
            return min(len(self.inedex_rgb), len(self.inedex_ir))
        return min(len(self.dataset_rgb), len(self.dataset_ir))

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname_rgb, pid_rgb, cpid_rgb, camid_rgb = self.dataset_rgb[self.inedex_rgb[index]]
        img_ori1 = _load_rgb(self.root_rgb, fname_rgb)

        if self.transform_rgb is not None:
            img1 = self.transform_rgb(img_ori1)

        fname_ir, pid_ir, cpid_ir, camid_ir = self.dataset_ir[self.inedex_ir[index]]
        img_ori2 = _load_rgb(self.root_ir, fname_ir)

        if self.transform_ir is not None:
            img2 = self.transform_ir(img_ori2)
        return img1, fname_rgb, pid_rgb, cpid_rgb, camid_rgb, self.inedex_rgb[index], \
            img2, fname_ir, pid_ir, cpid_ir, camid_ir, self.inedex_ir[index]
