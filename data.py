from PIL import Image as image
import torch
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import os
import torchvision.datasets as dset
voc_colormap = [[0, 0, 0], [245,222,179]]
class my_data(torch.utils.data.Dataset):
    #if target_transform=mask_transform is
    def __init__(self,root='/mnt/ramdisk',image_set='train',transform=None,target_transform=None):
        super(my_data,self).__init__()
        pth='/media/llm/D08E2AF4DD65FFFC/dataset/generate_sample_by_ios_image_size_256_256_thickness_0.2.csv'
        work_pth='/mnt/ramdisk'
        self.image_set=image_set
        f=[i.strip().split(',') for i in open(pth ).readlines()]
        if self.image_set=='train':
            self.file_name=f[:30000]
        else:
            self.file_name=f[37000:]
        self.image=[os.path.join(work_pth,x[0]) for x in self.file_name]
        self.mask=[os.path.join( work_pth,x[1]) for x in self.file_name]
        self.transform=transform
        self.target_transform=target_transform
        # self.root=os.path.expanduser(root)
        # self.transform=transform
        # self.target_transform=target_transform
        # self.image_set=image_set
        # voc_dir=os.path.join(self.root)
        # image_dir=os.path.join(voc_dir,'image')
        # mask_dir=os.path.join(voc_dir,'mask')
        # # splits_dir=os.path.join(voc_dir,'ImageSets/Segmentation')
        # splits_f=os.path.join(self.root, self.image_set + '.txt')
        # with open(os.path.join(splits_f),'r') as f:
        #     file_name=[x.strip() for x in f.readlines()]
        # self.image=[os.path.join(image_dir,x+'.jpg') for x in file_name]
        # self.mask=[os.path.join(mask_dir,x+'.png') for x in file_name]

        assert (len(self.image)==len(self.mask))

    def __getitem__(self, index):
        img=image.open(self.image[index]).convert('RGB')
        target=image.open(self.mask[index])  #binary set need only one
        return self.transform(img),self.target_transform(target)

    def __len__(self):
        return len(self.image)
class seg_target(object):
    def __init__(self):
        voc_colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
        self.class_index=np.zeros(256**3)
        for i,j in enumerate(voc_colormap):
            tmp=(j[0]*256+j[1])*256+j[2]
            self.class_index[tmp]=i
    def __call__(self, target ):
        target=target.convert('RGB')
        target=np.array(target).transpose(2,0,1).astype(np.int32)
        target=(target[0]*256+target[1])*256+target[2]
        target=self.class_index[target]
        return target
def hist(label_true,label_pred,num_cls):
    # mask=(label_true>=0)&(label_true<num_cls)
    hist=np.bincount(label_pred.astype(int)*num_cls+label_true.astype(int),minlength=num_cls**2).reshape(num_cls,num_cls)
    return hist
def label_acc_score(label_true,label_pred,num_cls):
    hist_matrix=np.zeros((num_cls,num_cls))
    tmp=0
    for i,j in zip(label_true,label_pred):
        hist_matrix+=hist(i.cpu().numpy().flatten(),j.cpu().numpy().flatten(),num_cls)
        tmp+=1
    diag=np.diag(hist_matrix)
    # acc=diag/hist_matrix.sum()
    acc_cls=diag/hist_matrix.sum(axis=0)
    m_iou=diag/(hist_matrix.sum(axis=1)+hist_matrix.sum(axis=0)-diag)
    return acc_cls,m_iou,hist_matrix,tmp