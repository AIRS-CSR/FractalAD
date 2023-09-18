import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
MVTec_DEFECT_TYPE = ['good', 'anomaly']

# max dataset size: 391(hazelnut)
# dataset size = 13 *32 = 416
dataset_size = 391

class MVTec(Dataset):
    def __init__(self, dataset_path='./Data/MVTec', class_name='bottle', is_train=True, load_size=256, transform_train=None):
        super().__init__()
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.load_size = load_size

        # load data path
        if is_train:
            self.images, self.labels, self.masks = self.load_train_folder()
        else:
            self.images, self.labels, self.masks = self.load_test_folder()

        if transform_train is not None:
            self.transform_train = transform_train
        else:
            self.transform_train = transforms.Compose([transforms.Resize((load_size, load_size)), transforms.ToTensor()])
        self.transform_test = transforms.Compose([transforms.Resize((load_size, load_size)), transforms.ToTensor()])
        self.transform_mask = transforms.Compose([transforms.Resize((load_size, load_size), Image.NEAREST), transforms.ToTensor()])
    
    def __getitem__(self, index):
        img, label, mask = self.images[index], self.labels[index], self.masks[
            index]
        # ../defect_type/000.png
        img_name = img.split('/')[-2] + '_'
        img_name += os.path.splitext(os.path.basename(img))[0]

        img = Image.open(img).convert('RGB')
        if self.is_train:
            # data augmentation on train data
            img, aug_img, aug_mask = self.transform_train(img)
            return img, aug_img, aug_mask, label
        else:
            img = self.transform_test(img)
            if MVTec_DEFECT_TYPE[label] == 'good':
                mask = torch.zeros([1, self.load_size, self.load_size])
            else:
                mask = Image.open(mask)
                mask = self.transform_mask(mask)
            return img, mask, label, img_name
    
    def __len__(self):
        return len(self.images)

    def load_train_folder(self):
        images, labels, masks = [], [], []
        img_dir = os.path.join(self.dataset_path, self.class_name, "train", "good")
        img_fpath_list = sorted([os.path.join(img_dir, f)
                                     for f in os.listdir(img_dir)
                                     if f.endswith('.png')])
        img_fpath_size = len(img_fpath_list)
        for idx in range(dataset_size):
            images.append(img_fpath_list[idx % img_fpath_size])
        labels.extend([0] * len(images))
        masks.extend([None] * len(images))
        assert len(images) == len(labels), 'number of x and y should be same'
        return list(images), list(labels), list(masks)

    def load_test_folder(self):
        images, labels, masks = [], [], []
        img_dir = os.path.join(self.dataset_path, self.class_name, "test")
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])
            images.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                labels.extend([0] * len(img_fpath_list))
                masks.extend([None] * len(img_fpath_list))
            else:
                labels.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                masks.extend(gt_fpath_list)

        assert len(images) == len(labels), 'number of x and y should be same'

        return list(images), list(labels), list(masks)