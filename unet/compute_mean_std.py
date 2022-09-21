import os
from PIL import Image
import numpy as np


def main():
    img_channels = 1
    root = 'train_img_label.txt'
    with open(root, 'r') as f:
        img_name_list = f.read().splitlines()
    img_name_list = [i.split(' ')[0] for i in img_name_list]
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)
    for img_path in img_name_list:
        img = np.array(Image.open(img_path)) / 255.

        cumulative_mean += img.mean()
        cumulative_std += img.std()

    mean = cumulative_mean / len(img_name_list)
    std = cumulative_std / len(img_name_list)
    print(f"mean: {mean}")
    print(f"std: {std}")


if __name__ == '__main__':
    main()
