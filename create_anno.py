import os

data_root = './COVID-19_Dataset'
train_ratio = 0.7

covid_path = os.path.join(data_root, 'COVID')
normal_path = os.path.join(data_root, 'Normal')

covid_img_path = os.path.join(covid_path, 'images')
covid_mask_path = os.path.join(covid_path, 'masks')
normal_img_path = os.path.join(normal_path, 'images')
normal_mask_path = os.path.join(normal_path, 'masks')

# deal imgs
covid_len = len(os.listdir(covid_img_path))
normal_len = len(os.listdir(normal_img_path))

for i, name in enumerate(os.listdir(covid_img_path)):
    if i <= train_ratio * covid_len:
        with open('train_img_label.txt', 'a') as f:
            f.write(os.path.join(covid_img_path, name) + ' ' + '1\n')
        with open('train_mask.txt', 'a') as f:
            f.write(os.path.join(covid_mask_path, name) + '\n')
    else:
        with open('val_img_label.txt', 'a') as f:
            f.write(os.path.join(covid_img_path, name) + ' ' + '1\n')
        with open('val_mask.txt', 'a') as f:
            f.write(os.path.join(covid_mask_path, name) + '\n')

for i, name in enumerate(os.listdir(normal_img_path)):
    if i <= train_ratio * normal_len:
        with open('train_img_label.txt', 'a') as f:
            f.write(os.path.join(normal_img_path, name) + ' ' + '0\n')
        with open('train_mask.txt', 'a') as f:
            f.write(os.path.join(normal_mask_path, name) + '\n')
    else:
        with open('val_img_label.txt', 'a') as f:
            f.write(os.path.join(normal_img_path, name) + ' ' + '0\n')
        with open('val_mask.txt', 'a') as f:
            f.write(os.path.join(normal_mask_path, name) + '\n')





