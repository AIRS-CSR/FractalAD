import torch
import numpy as np
from PIL import Image, ImageChops
import cv2
import random
from torchvision import transforms
from fractals import ifs


class FAG(object):
    def __init__(self, load_size, lb=0.1, hb=0.5, rotation=[15,75], colorJitter=0.2, noise_amp=10,
                start_rito=0.1, end_rito=0.9, is_findobj=False):
        self.load_size, self.lb, self.hb, self.rotation, self.colorJitter, \
        self.noise_amp, self.start_rito, self.end_rito, self.is_findobj \
        = \
        load_size, lb, hb, rotation, colorJitter, \
        noise_amp, start_rito, end_rito, is_findobj

    def OBJ_mask(self, img):
        w, h = img.size
        img = np.array(img)
        img = cv2.resize(img, (256,256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.blur(img, (5,5))
        mean = cv2.mean(img[0:20,0:20])
        if mean[0]>128:
            _, obj_mask = cv2.threshold(img, int(mean[0])-50, 255, cv2.THRESH_BINARY_INV)
        else:
            _, obj_mask = cv2.threshold(img, int(mean[0])+10, 255, cv2.THRESH_BINARY)
        # _, bn = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        obj_mask = cv2.medianBlur(obj_mask, 5)
        obj_mask = cv2.resize(obj_mask, (w,h))
        # cv2.imwritoe('kkk.jpg', bn)
        
        cnts, _ = cv2.findContours(obj_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)

        return obj_mask, rect[1]

    def coord_in_mask(self, mask, h, w, cut_size):
        flag = 1
        while flag:
            y = int(random.uniform(cut_size//2, h-cut_size//2))
            x = int(random.uniform(cut_size//2, w-cut_size//2))
            if mask[y, x]>0:
                flag = 0

        return y-cut_size//2, x-cut_size//2

    def ChangeColor(self, img):
        r, g, b = img.split()
        chs = [r, g, b]
        c_c = np.random.choice([0,1,2],2)
        r_c = np.random.random([2])
        for i in range(2):
            chs[c_c[i]] = Image.fromarray(np.uint8(chs[c_c[i]] * r_c[i] * 2))
        img = Image.merge('RGB',(chs))

        return img

    def AddGaussianNoise(self, img, amplitude, mask):
        img = np.array(img)
        h, w, c = img.shape
        N = amplitude * np.random.normal(loc=0, scale=1, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        tmp = img + N
        tmp[tmp>255] = 255
        tmp[tmp<0] = 0
        mask = np.uint8(np.array(mask))
        mask = np.reshape(mask, [h, w, 1])
        mask = np.repeat(mask, c, axis=2)
        img = cv2.bitwise_and(np.uint8(img), 255-mask) + cv2.bitwise_and(np.uint8(tmp), mask)
        img = Image.fromarray(img)
        return img

    def Jitter(self, img, colorJitter):
        Random_Jitter = []
        Random_Jitter.append(transforms.ColorJitter(brightness = colorJitter,
                                                contrast = colorJitter,
                                                saturation = colorJitter,
                                                hue = colorJitter))
        Random_Jitter.append(transforms.ColorJitter(brightness = 1, contrast = 1))
        Random_Jitter.append(transforms.Lambda(self.ChangeColor))
        Jitter_transforms = transforms.RandomChoice(Random_Jitter)
        ColorJitter_img = Jitter_transforms(img)

        return ColorJitter_img

    def __call__(self, img):

        w, h = img.size
        proc_size = w if w<h else h
        # cut region    
        cut_size = int(random.uniform(self.lb*proc_size, self.hb*proc_size))

        if self.is_findobj:
            obj_mask, rect = self.OBJ_mask(img)
            proc_size = rect[0] if rect[0]<rect[1] else rect[1]
            cut_size = int(random.uniform(0.5*self.lb*proc_size, 3*self.hb*proc_size))
            from_location_h, from_location_w = self.coord_in_mask(obj_mask, h, w, cut_size)
            box = [from_location_w, from_location_h, from_location_w + cut_size, from_location_h + cut_size]
        else:
            h_start, w_start, h_end, w_end = h*self.start_rito, w*self.start_rito, h*self.end_rito, w*self.end_rito
            from_location_h = int(random.uniform(h_start, h_end - cut_size))
            from_location_w = int(random.uniform(w_start, w_end - cut_size))
            box = [from_location_w, from_location_h, from_location_w + cut_size, from_location_h + cut_size]
        
        patch = img.crop(box)

        # colorJitter
        patch = self.Jitter(patch, self.colorJitter)
        
        # flip
        if np.random.random()>0.5:
            patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.random()>0.5:
            patch = patch.transpose(Image.FLIP_TOP_BOTTOM)

        # rotate
        rot_deg = random.uniform(*self.rotation)
        patch = patch.convert("RGBA").rotate(rot_deg, expand=False)
        
        # target
        if self.is_findobj:
            to_location_h, to_location_w = self.coord_in_mask(obj_mask, h, w, cut_size)
        else:
            h_start, w_start, h_end, w_end = h*self.start_rito, w*self.start_rito, h*self.end_rito, w*self.end_rito
            to_location_h = int(random.uniform(h_start, h_end - cut_size))
            to_location_w = int(random.uniform(w_start, w_end - cut_size))

        mask = patch.split()[-1]
        patch = patch.convert("RGB")

        # render fractals in binary
        system = ifs.sample_system(2)
        points = ifs.iterate(system, 100000)
        fractal_img = ifs.render(points, s=cut_size, binary=True, patch=True)
        fractal_img = Image.fromarray(255*fractal_img)
        mask = ImageChops.multiply(mask, fractal_img)
        patch = ImageChops.multiply(patch, fractal_img.convert('RGB'))
        
        # paste
        aug = img.copy()
        aug.paste(patch, (to_location_w, to_location_h), mask=mask)

        # get msk
        msk = Image.new('L', img.size)
        ones_patch = Image.new('L', patch.size, 255)
        msk.paste(ones_patch, (to_location_w, to_location_h), mask=mask)
        msk = msk.resize((self.load_size, self.load_size), Image.NEAREST)
        img = img.resize((self.load_size, self.load_size), Image.ANTIALIAS)
        aug = aug.resize((self.load_size, self.load_size), Image.ANTIALIAS)
        msk = Image.fromarray(cv2.dilate(np.array(msk), np.ones((3,3),np.uint8), iterations = 3))
        # Add Gaussian Noise
        if np.random.random()>0.5:
            aug = self.AddGaussianNoise(aug, amplitude=self.noise_amp, mask=msk) 
        color_jitter = transforms.ColorJitter(brightness = 0.05, contrast = 0.05, saturation = 0.05, hue = 0.05)
        seed = np.random.choice(range(100000),1)
        torch.manual_seed(seed)
        img = color_jitter(img)
        torch.manual_seed(seed)
        aug = color_jitter(aug)

        img = transforms.ToTensor()(img)
        aug = transforms.ToTensor()(aug)
        msk = transforms.ToTensor()(msk)

        return img, aug, msk