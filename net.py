import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import torchvision as tv
import random
import math
from functools import reduce
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader as dataloader
# import nonechucks as nc
from data import my_data,label_acc_score,voc_colormap,seg_target
from MobileNet import MobileNetV2
vgg=tv.models.vgg13_bn(pretrained=True)
image_transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
mask_transform=transforms.Compose([transforms.ToTensor()])
trainset=my_data(transform=image_transform,target_transform=mask_transform)
testset=my_data(image_set='test',transform=image_transform,target_transform=mask_transform)
trainload=torch.utils.data.DataLoader(trainset,batch_size=32)
testload=torch.utils.data.DataLoader(testset,batch_size=32)
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device=torch.device('cpu')
print (device)
net=MobileNetV2()
net.load_state_dict(torch.load('mobilenet_v2.pth.tar'))
def weights_init(m):
    pass
class Mobile(nn.Module):
    def __init__(self):
        super(Mobile,self).__init__()
        self.l0_0=vgg.features[:6]
        self.l1_0 = net.features[0:1]     # 32
        self.l2_0 = net.features[1:4]   # 24
        self.l3_0 = net.features[4:7]   # 32
        self.l4_0 = net.features[7:14]  # 96
        self.sideout0=nn.Conv2d(64,1,1)
        self.u1=nn.Conv2d(32,1,1)
        self.u2=nn.Conv2d(24,1,1)
        self.u3=nn.Conv2d(32,1,1)
        self.u4=nn.Conv2d(96,1,1)
        self.sideout1=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sideout2=nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.sideout3=nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.sideout4=nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.fuse=nn.Conv2d(5,1,1,)
        self.first_layer=nn.Sequential(
            nn.Conv2d(64,3,1),
            nn.BatchNorm2d(3),
            nn.ReLU(True))
    def forward(self, input):
        l0_0=self.l0_0(input)
        l1_0=self.l1_0(self.first_layer(l0_0))
        l2_0=self.l2_0(l1_0)
        l3_0=self.l3_0(l2_0)
        l4_0=self.l4_0(l3_0)
        u_1=self.u1(l1_0)
        u_2=self.u2(l2_0)
        u_3=self.u3(l3_0)
        u_4=self.u4(l4_0)
        sideout0=self.sideout0(l0_0)
        sideout1=self.sideout1(u_1)
        sideout2=self.sideout2(u_2)
        sideout3=self.sideout3(u_3)
        sideout4=self.sideout4(u_4)
        fuse=self.fuse(torch.cat([sideout0,sideout1,sideout2,sideout3,sideout4],dim=1))
        return sideout0,sideout1,sideout2,sideout3,sideout4,fuse
class bilance_cross(nn.Module):
    def __init__(self):
        super(bilance_cross,self).__init__()
    # def b_loss(self):
    #     self.sig=nn.LogSigmoid()
    #     self.cross_entropy=nn.NLLLoss(torch.tensor([0.1]))
    def forward(self, pred,target):
        a,b=torch.bincount(target.int().reshape([-1]))
        self.sig=nn.LogSigmoid()
        self.cross_entropy=nn.BCELoss(a.float()/b.float())
        return self.cross_entropy(self.sig(pred),target)
dtype=torch.float
def train(epoch):
    model=Mobile()
    model.train()
    model=model.to(device)
    # side_loss=bilance_cross()
    # fuse_loss=nn.CrossEntropyLoss()
    optimize=torch.optim.Adam(model.parameters(),lr=0.001)
    store_loss=[]
    for i in range(epoch):
        tmp=0
        for image,mask in trainload:
            image,mask=image.to(device,dtype=dtype),mask.to(device,dtype=dtype)
            a,b=torch.bincount(mask.int().reshape([-1]))
            beta=b.float()/(a+b)
            loss = nn.BCELoss(beta/(1-beta))
            optimize.zero_grad()
            l0,l1,l2,l3,l4,l=model(image)
            loss_list=list(map(lambda x,y:side_loss(x,y),[l0,l1,l2,l3,l4,],[mask]*5))
            tmp=reduce(lambda x,y:x+y,loss_list)
            loss=tmp/5+fuse_loss(l,mask)
            loss.backward()
            optimize.step()
            tmp=loss.data
            # print ("loss ",tmp)
            # break
        store_loss.append(tmp)
        print ("{0} epoch ,loss is {1}".format(i,tmp))
    return model,store_loss
def torch_pic(img,pred,mask):
    img, pred, mask = img[:4], pred[:4].to(torch.long), mask[:4].to(torch.long)
    pred = pred.squeeze(dim=1)
    mask = mask.squeeze(dim=1)
    voc_colormap = [[0, 0, 0], [245, 222, 179]]
    voc_colormap = torch.from_numpy(np.array(voc_colormap))
    voc_colormap = voc_colormap.to(dtype)
    mean, std = np.array((0.485, 0.456, 0.406)), np.array((0.229, 0.224, 0.225))
    mean, std = torch.from_numpy(mean).to(dtype), torch.from_numpy(std).to(dtype)
    img = img.permute(0, 2, 3, 1)
    img = (img * std + mean)
    # pred=pred.permute(0,2,3,1)
    # mask=mask.permute(0,2,3,1)
    pred = voc_colormap[pred] / 255.0
    mask = voc_colormap[mask] / 255.0
    pred=pred.permute(0, 3, 1, 2)
    mask=mask.permute(0, 3, 1, 2)
    tmp = tv.utils.make_grid(torch.cat((img.permute(0,3,1,2), pred, mask)), nrow=4)
    plt.imshow(tmp.permute(1,2,0))
    plt.show()


def test(model):
    img=[]
    pred=[]
    mask=[]
    l0_list = []
    l1_list=[]
    l2_list=[]
    l3_list=[]
    l4_list=[]
    l5_list=[]
    with torch.no_grad():
        model.eval()
        model.to(device)
        for image,mask_img in testload:
            image=image.to(device,dtype=dtype)
            l0,l1,l2,l3,l4,l5=model(image)
            label=l5.cpu()>0.5
            # label=((l0+l1+l2+l3+l4)/5.)>0
            # l1_list.append((l1>0.5).to(torch.long))
            # l2_list.append((l2>0.5).to(torch.long))
            # l3_list.append((l3>0.5).to(torch.long))
            # l4_list.append((l4>0.5).to(torch.long))
            l0_list.append(l0)
            l1_list.append(l1)
            l2_list.append(l2)
            l3_list.append(l3)
            l4_list.append(l4)
            l5_list.append(l5)



            pred.append(label.to(torch.long))
            img.append(image.cpu())
            mask.append(mask_img)
    return torch.cat(img),torch.cat(pred),torch.cat(mask),[l1_list,l2_list,l3_list,l4_list,l5_list]
epoch=5
model=Mobile()
model.train()
model=model.to(device)
side_loss=bilance_cross()
fuse_loss=nn.CrossEntropyLoss()
optimize=torch.optim.Adam(model.parameters(),lr=0.001)
store_loss=[]
for i in range(epoch):
    tmp=0
    for image,mask in trainload:
        image,mask=image.to(device,dtype=dtype),mask.to(device,dtype=dtype)
        a, b = torch.bincount(mask.int().reshape([-1]))
        beta = a.float() / (a + b)
        crition=nn.BCEWithLogitsLoss(weight=beta/(1-beta),pos_weight=beta/(1-beta))
        optimize.zero_grad()
        l0,l1,l2,l3,l4,l=model(image)
        loss_list=list(map(lambda x,y:crition(x,y),[l0,l1,l2,l3,l4,],[mask]*5))
        tmp=reduce(lambda x,y:x+y,loss_list)
        loss=tmp/5+crition(l,mask)
        loss.backward()
        optimize.step()
        tmp=loss.data
        print ("loss ",tmp)
        # break
    store_loss.append(tmp)
    print ("{0} epoch ,loss is {1}".format(i,tmp))


torch.save(model.state_dict(),'mobile_beta')

plt.rcParams['figure.dpi'] = 300
#
# model=Mobile()
# model.load_state_dict(torch.load('mobile'))
img,pred,mask,l=test(model)
torch_pic(img[20:24],pred[20:24].to(torch.long),mask[20:24].to(torch.long))