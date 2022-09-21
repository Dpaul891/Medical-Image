import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(MyDataset, self).__init__()
        self.flag = "train" if train else "val"
        img_txt = self.flag + '_img_label.txt'
        mask_txt = self.flag + '_mask.txt'
        self.transforms = transforms
        
        with open(os.path.join(root, img_txt), 'r') as f:
            self.img_list = f.read().splitlines()
            self.img_list = [i.split(' ')[0] for i in self.img_list]
        with open(os.path.join(root, mask_txt), 'r') as f:
            self.gt_list = f.read().splitlines()
        
    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        gt = Image.open(self.gt_list[idx])
        gt = np.array(gt) / 255
        gt = gt[:, :, 0]
        mask = Image.fromarray(gt)
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        
        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

