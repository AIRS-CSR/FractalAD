import argparse
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, auc
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mvtec import *
from fag import *
from networks import *
from tools import *

torch.cuda.empty_cache()

class FractalAD(): 
    def __init__(self):
        self.findobj_cfg = {'bottle':0,'cable':0,'capsule':1,'carpet':0,'grid':0,
        'hazelnut':0,'leather':0,'metal_nut':0,'pill':1,'screw':1,'tile':0,
        'toothbrush':1,'transistor':0,'wood':0,'zipper':1}
        self.kd_cfg = {'bottle':1,'cable':1,'capsule':1,'carpet':0,'grid':0,
        'hazelnut':1,'leather':0,'metal_nut':1,'pill':1,'screw':1,'tile':0,
        'toothbrush':1,'transistor':1,'wood':0,'zipper':1}
        self.load_model()

    def load_dataset(self, load_size, batch_size, num_workers=0, drop_last=False, is_train=True):
        if is_train:
            transform_train = FAG(load_size=load_size, is_findobj=self.findobj_cfg[category])
        else:
            transform_train = None
        dataset = MVTec(dataset_path, category, is_train, load_size, transform_train)  
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train,
                                num_workers=num_workers, drop_last=drop_last, pin_memory=True)
        phase = 'train' if is_train else 'test'
        self.dataset_length = len(dataset)
        print('Dataset size : {} set - {}'.format(phase, self.dataset_length))
        return dataloader

    def load_model(self):
        self.teacher = BackBone(backbone=backbone, pretrained=True).to(device) 
        self.student = BackBone(backbone=backbone, pretrained=False).to(device)
        self.sfnet = SFNet(backbone=backbone).to(device)
        self.csam = CSAM()
        if self.kd_cfg[category]:
            self.optimizer = torch.optim.AdamW([{'params':self.student.parameters()}, 
                                                {'params':self.sfnet.parameters()}], lr=lr, weight_decay=0.001)
        else:
            self.optimizer = torch.optim.AdamW(self.sfnet.parameters(), lr=lr)
        self.db_loss = DiceBCELoss()
        self.kd_loss = CosSimLoss()

    def train_step(self, img, aug, msk):
        self.optimizer.zero_grad()

        if self.kd_cfg[category]:

            img_t = self.teacher(img)
            aug_t = self.teacher(aug)
            img_s = self.student(img)
            aug_s = self.student(aug)
            prd = self.sfnet(self.csam(aug_t, aug_s))

            loss_kd = self.kd_loss(img_t, img_s)
            loss_db = self.db_loss(prd, msk)
            loss = loss_kd + loss_db

            loss.backward()
            self.optimizer.step()

            return torch.sigmoid(prd), [loss_kd, loss_db, loss]

        else:

            prd = self.sfnet(self.teacher(aug))
            loss = self.db_loss(prd, msk)

            loss.backward()
            self.optimizer.step()

            return torch.sigmoid(prd), loss

    def train(self):
        print('Training category: ', category)
        print('Training num_epoch: ', num_epoch)
        train_loader = self.load_dataset(load_size = load_size, batch_size=batch_size,
                                         num_workers=8, drop_last=True, is_train=True)

        self.teacher.eval()
        self.student.train()
        self.sfnet.train()
        
        start_epoch = 0
        global_step = 0

        # load weights
        path_checkpoint = os.path.join(weight_save_path, "model.pth")
        if os.path.exists(path_checkpoint):
            checkpoint = torch.load(path_checkpoint)
            if self.kd_cfg[category]:
                self.student.load_state_dict(checkpoint['student'])
            self.sfnet.load_state_dict(checkpoint['sfnet'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            global_step = checkpoint['global_step']
            print('-'*50)
            print("Model restored")
            print('-'*50)

        for epoch in range(start_epoch, num_epoch):
            start_time = time.time()

            for idx, (img, aug, msk, _) in enumerate(train_loader):
                global_step += 1
                img = img.to(device)
                aug = aug.to(device)
                msk = msk.to(device)
                prd, loss = self.train_step(img, aug, msk)

                # tensorboard
                if self.kd_cfg[category]:
                    loss_kd, loss_db, loss = loss

            # save images
            img = np.uint8(255*img[0,:,:,:].permute(1,2,0).to('cpu').detach().numpy())
            aug = np.uint8(255*aug[0,:,:,:].permute(1,2,0).to('cpu').detach().numpy())
            msk = np.uint8(255*msk[0,:,:,:].permute(1,2,0).to('cpu').detach().numpy())
            prd = np.uint8(255*prd[0,:,:,:].permute(1,2,0).to('cpu').detach().numpy())

            msks = cv2.cvtColor(np.concatenate([msk, prd], axis=1), cv2.COLOR_GRAY2RGB)
            imgs = cv2.cvtColor(np.concatenate([img, aug, msks], axis=1), cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(train_path, 'epoch_{:04d}.jpg'.format(epoch)), imgs)

            # save weights
            checkpoint = {
                "sfnet": self.sfnet.state_dict(),
                'optimizer':self.optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step
            }
            if self.kd_cfg[category]:
                checkpoint.update({"student": self.student.state_dict()})
            torch.save(checkpoint, os.path.join(weight_save_path, 'model.pth'))  

            if self.kd_cfg[category]:
                print('Epoch: {}/{} | loss_kd: {:.4f} | loss_db: {:.4f} | loss: {:.4f} | Time consumed: {:.4f}'.\
                format(epoch, num_epoch, float(loss_kd.data), float(loss_db.data), float(loss.data), time.time() - start_time))
            else:
                print('Epoch: {}/{} | loss: {:.4f} | Time consumed: {:.4f}'.\
                format(epoch, num_epoch, float(loss.data), time.time() - start_time))
        print('Train end.')

    def test(self):
        print('Testing category: ', category)
        try:
            if self.kd_cfg[category]:
                self.student.load_state_dict(torch.load(weight_save_path+'/model.pth')['student'])
            self.sfnet.load_state_dict(torch.load(weight_save_path+'/model.pth')['sfnet'])
        except:
            raise Exception('Check saved model path.')
            
        self.teacher.eval()
        self.student.eval()
        self.sfnet.eval()

        test_loader = self.load_dataset(load_size = load_size, batch_size=1, is_train=False)

        # prepare for calculating auc-pro
        gt_masks = []
        pd_masks = []
        gt_list_img = []
        pd_list_img = []
        gt_list_pix = []
        pd_list_pix = []

        start_time = time.time()
        for test_img, test_msk, label, img_name in tqdm(test_loader):
            # test image shape = [1,3,H,W]
            test_img = test_img.to(device)
            with torch.set_grad_enabled(False):
                if self.kd_cfg[category]:
                    pred_msk = torch.sigmoid(self.sfnet(self.csam(self.teacher(test_img), self.student(test_img)))).to('cpu')
                else:
                    pred_msk = torch.sigmoid(self.sfnet(self.teacher(test_img))).to('cpu')
            gt_masks.append(test_msk.squeeze().numpy())
            pd_masks.append(pred_msk.squeeze().numpy())
            # test image label 0 is normal, 1 is anomaly
            gt_score = 0. if MVTec_DEFECT_TYPE[label] == 'good' else 1.
            pd_score = torch.mean(pred_msk).item()
            gt_list_img.append(gt_score)
            pd_list_img.append(pd_score)
            gt_list_pix.extend(test_msk.numpy().ravel())
            pd_list_pix.extend(pred_msk.numpy().ravel())

            # save images
            test_img = np.uint8(255 * test_img[0, :, :, :].permute(1, 2, 0).to('cpu').numpy())
            test_msk = np.uint8(255 * test_msk[0, :, :, :].permute(1, 2, 0).numpy())
            pred_msk = np.uint8(255 * pred_msk[0, :, :, :].permute(1, 2, 0).numpy())
            msks = cv2.cvtColor(np.concatenate([test_msk, pred_msk], axis=1), cv2.COLOR_GRAY2RGB)
            imgs = cv2.cvtColor(np.concatenate([test_img, msks], axis=1), cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(sample_path, f'{img_name[0]}.jpg'), imgs)
                
        print('Total test time consumed : {}'.format(time.time() - start_time))

        # calculate auc-roc
        p_auc = round(roc_auc_score(gt_list_pix, pd_list_pix), 4)
        i_auc = round(roc_auc_score(gt_list_img, pd_list_img), 4)

        print("Total pixel-level auc-roc score:", p_auc)
        print("Total image-level auc-roc score:", i_auc)

        with open(os.path.join(project_path, backbone, 'results.txt'), 'a', encoding='utf-8') as f:
            f.write(category + ' pixel-auc:' + str(p_auc) + ' image-auc:' + str(i_auc) + '\n')
        print('Test end.')

def get_args():
    parser = argparse.ArgumentParser(description='FractalAD')
    parser.add_argument('--phase', default='train')
    parser.add_argument('--dataset_path', default=r'./MVTec_AD/') 
    parser.add_argument('--backbone', default='resnet18')
    parser.add_argument('--category', default='capsule')
    parser.add_argument('--num_epoch', default=100)
    parser.add_argument('--lr', default=0.001)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--load_size', default=256)
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--project_path', default='./results') 
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    seed_everything(111111)
    args = get_args()
    torch.cuda.set_device(int(args.gpu))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print ('Available devices ', torch.cuda.device_count())
    print ('Current cuda device ', torch.cuda.current_device())
    # print(torch.cuda.get_device_name(device))
    
    phase = args.phase
    dataset_path = args.dataset_path
    backbone = args.backbone
    category = args.category
    num_epoch = int(args.num_epoch)
    lr = args.lr
    batch_size = args.batch_size
    load_size = args.load_size
    project_path = args.project_path
    train_path = os.path.join(project_path, backbone, 'train', category + '_bs'+ str(batch_size))
    os.makedirs(train_path, exist_ok=True)    
    sample_path = os.path.join(project_path, backbone, 'sample', category + '_bs'+ str(batch_size))
    os.makedirs(sample_path, exist_ok=True)
    weight_save_path = os.path.join(project_path, backbone, 'checkpoint', category + '_bs'+ str(batch_size))
    os.makedirs(weight_save_path, exist_ok=True)

    fractalad = FractalAD()
    if phase == 'train':
        fractalad.train()
        # fractalad.test()
    elif phase == 'test':
        fractalad.test()
    else:
        print('Phase argument must be train or test.')
