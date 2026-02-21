#################################################
# C:\Dan_WS\project_src\SelfBlendedImages-master\SelfBlendedImages-master\src\utils\__init__.py
#################################################

#################################################
# C:\Dan_WS\project_src\SelfBlendedImages-master\SelfBlendedImages-master\src\utils\blend.py
#################################################
# Created by: Kaede Shiohara
# Yamasaki Lab at The University of Tokyo
# shiohara@cvm.t.u-tokyo.ac.jp
# Copyright (c) 2021
# 3rd party softwares' licenses are noticed at https://github.com/mapooon/SelfBlendedImages/blob/master/LICENSE

import cv2
import numpy as np
import scipy as sp
from skimage.measure import label, regionprops
import random
from PIL import Image
import sys



def alpha_blend(source,target,mask):
	mask_blured = get_blend_mask(mask)
	img_blended=(mask_blured * source + (1 - mask_blured) * target)
	return img_blended,mask_blured

def dynamic_blend(source,target,mask):
	mask_blured = get_blend_mask(mask)
	blend_list=[0.25,0.5,0.75,1,1,1]
	blend_ratio = blend_list[np.random.randint(len(blend_list))]
	mask_blured*=blend_ratio
	img_blended=(mask_blured * source + (1 - mask_blured) * target)
	return img_blended,mask_blured

def get_blend_mask(mask):
	H,W=mask.shape
	size_h=np.random.randint(192,257)
	size_w=np.random.randint(192,257)
	mask=cv2.resize(mask,(size_w,size_h))
	kernel_1=random.randrange(5,26,2)
	kernel_1=(kernel_1,kernel_1)
	kernel_2=random.randrange(5,26,2)
	kernel_2=(kernel_2,kernel_2)
	
	mask_blured = cv2.GaussianBlur(mask, kernel_1, 0)
	mask_blured = mask_blured/(mask_blured.max())
	mask_blured[mask_blured<1]=0
	
	mask_blured = cv2.GaussianBlur(mask_blured, kernel_2, np.random.randint(5,46))
	mask_blured = mask_blured/(mask_blured.max())
	mask_blured = cv2.resize(mask_blured,(W,H))
	return mask_blured.reshape((mask_blured.shape+(1,)))


def get_alpha_blend_mask(mask):
	kernel_list=[(11,11),(9,9),(7,7),(5,5),(3,3)]
	blend_list=[0.25,0.5,0.75]
	kernel_idxs=random.choices(range(len(kernel_list)), k=2)
	blend_ratio = blend_list[random.sample(range(len(blend_list)), 1)[0]]
	mask_blured = cv2.GaussianBlur(mask, kernel_list[0], 0)
	# print(mask_blured.max())
	mask_blured[mask_blured<mask_blured.max()]=0
	mask_blured[mask_blured>0]=1
	# mask_blured = mask
	mask_blured = cv2.GaussianBlur(mask_blured, kernel_list[kernel_idxs[1]], 0)
	mask_blured = mask_blured/(mask_blured.max())
	return mask_blured.reshape((mask_blured.shape+(1,)))



#################################################
# C:\Dan_WS\project_src\SelfBlendedImages-master\SelfBlendedImages-master\src\utils\funcs.py
#################################################

import sys
import json
import numpy as np
from PIL import Image
from glob import glob 
import os
import pandas as pd
import albumentations as alb
import cv2

def load_json(path):
	d = {}
	with open(path, mode="r") as f:
		d = json.load(f)
	return d


def IoUfrom2bboxes(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou



def crop_face(img,landmark=None,bbox=None,margin=False,crop_by_bbox=True,abs_coord=False,only_img=False,phase='train'):
	assert phase in ['train','val','test']

	#crop face------------------------------------------
	H,W=len(img),len(img[0])

	assert landmark is not None or bbox is not None

	H,W=len(img),len(img[0])
	
	if crop_by_bbox:
		x0,y0=bbox[0]
		x1,y1=bbox[1]
		w=x1-x0
		h=y1-y0
		w0_margin=w/4#0#np.random.rand()*(w/8)
		w1_margin=w/4
		h0_margin=h/4#0#np.random.rand()*(h/5)
		h1_margin=h/4
	else:
		x0,y0=landmark[:68,0].min(),landmark[:68,1].min()
		x1,y1=landmark[:68,0].max(),landmark[:68,1].max()
		w=x1-x0
		h=y1-y0
		w0_margin=w/8#0#np.random.rand()*(w/8)
		w1_margin=w/8
		h0_margin=h/2#0#np.random.rand()*(h/5)
		h1_margin=h/5

	

	if margin:
		w0_margin*=4
		w1_margin*=4
		h0_margin*=2
		h1_margin*=2
	elif phase=='train':
		w0_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
		w1_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
		h0_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
		h1_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()	
	else:
		w0_margin*=0.5
		w1_margin*=0.5
		h0_margin*=0.5
		h1_margin*=0.5
			
	y0_new=max(0,int(y0-h0_margin))
	y1_new=min(H,int(y1+h1_margin)+1)
	x0_new=max(0,int(x0-w0_margin))
	x1_new=min(W,int(x1+w1_margin)+1)
	
	img_cropped=img[y0_new:y1_new,x0_new:x1_new]
	if landmark is not None:
		landmark_cropped=np.zeros_like(landmark)
		for i,(p,q) in enumerate(landmark):
			landmark_cropped[i]=[p-x0_new,q-y0_new]
	else:
		landmark_cropped=None
	if bbox is not None:
		bbox_cropped=np.zeros_like(bbox)
		for i,(p,q) in enumerate(bbox):
			bbox_cropped[i]=[p-x0_new,q-y0_new]
	else:
		bbox_cropped=None

	if only_img:
		return img_cropped
	if abs_coord:
		return img_cropped,landmark_cropped,bbox_cropped,(y0-y0_new,x0-x0_new,y1_new-y1,x1_new-x1),y0_new,y1_new,x0_new,x1_new
	else:
		return img_cropped,landmark_cropped,bbox_cropped,(y0-y0_new,x0-x0_new,y1_new-y1,x1_new-x1)


class RandomDownScale(alb.core.transforms_interface.ImageOnlyTransform):
	def apply(self,img,**params):
		return self.randomdownscale(img)

	def randomdownscale(self,img):
		keep_ratio=True
		keep_input_shape=True
		H,W,C=img.shape
		ratio_list=[2,4]
		r=ratio_list[np.random.randint(len(ratio_list))]
		img_ds=cv2.resize(img,(int(W/r),int(H/r)),interpolation=cv2.INTER_NEAREST)
		if keep_input_shape:
			img_ds=cv2.resize(img_ds,(W,H),interpolation=cv2.INTER_LINEAR)

		return img_ds
#################################################
# C:\Dan_WS\project_src\SelfBlendedImages-master\SelfBlendedImages-master\src\utils\initialize.py
#################################################
from glob import glob
import os
import sys
import json
import numpy as np
from PIL import Image
from glob import glob 
import os
import pandas as pd


def init_ff(phase,level='frame',n_frames=8):
	dataset_path='data/FaceForensics++/original_sequences/youtube/raw/frames/'
	

	image_list=[]
	label_list=[]

	
	
	folder_list = sorted(glob(dataset_path+'*'))
	filelist = []
	list_dict = json.load(open(f'data/FaceForensics++/{phase}.json','r'))
	for i in list_dict:
		filelist+=i
	folder_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]

	if level =='video':
		label_list=[0]*len(folder_list)
		return folder_list,label_list
	for i in range(len(folder_list)):
		# images_temp=sorted([glob(folder_list[i]+'/*.png')[0]])
		images_temp=sorted(glob(folder_list[i]+'/*.png'))
		if n_frames<len(images_temp):
			images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
		image_list+=images_temp
		label_list+=[0]*len(images_temp)

	return image_list,label_list




#################################################
# C:\Dan_WS\project_src\SelfBlendedImages-master\SelfBlendedImages-master\src\utils\logs.py
#################################################
import os
import logging

# a function  to create and save logs in the log files
def log(path, file):
    """[Create a log file to record the experiment's logs]
    
    Arguments:
        path {string} -- path to the directory
        file {string} -- file name
    
    Returns:
        [obj] -- [logger that record logs]
    """

    # check if the file exist
    log_file = os.path.join(path, file)

    if not os.path.isfile(log_file):
        open(log_file, "w+").close()

    # console_logging_format = "%(levelname)s %(message)s"
    # file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"
    console_logging_format = "%(message)s"
    file_logging_format = "%(message)s"
    # configure logger
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()
    
    # create a file handler for output file
    handler = logging.FileHandler(log_file)

    # set the logging level for log file
    handler.setLevel(logging.INFO)
    
    # create a logging format
    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    return logger

#################################################
# C:\Dan_WS\project_src\SelfBlendedImages-master\SelfBlendedImages-master\src\utils\sam.py
#################################################
# borrowed from 

import torch

import torch
import torch.nn as nn

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, nn.BatchNorm2d):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, nn.BatchNorm2d) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

#################################################
# C:\Dan_WS\project_src\SelfBlendedImages-master\SelfBlendedImages-master\src\utils\sbi.py
#################################################

# Created by: Kaede Shiohara
# Yamasaki Lab at The University of Tokyo
# shiohara@cvm.t.u-tokyo.ac.jp
# Copyright (c) 2021
# 3rd party softwares' licenses are noticed at https://github.com/mapooon/SelfBlendedImages/blob/master/LICENSE

import torch
from torchvision import datasets,transforms,utils
from torch.utils.data import Dataset,IterableDataset
from glob import glob
import os
import numpy as np
from PIL import Image
import random
import cv2
from torch import nn
import sys
import albumentations as alb

import warnings
warnings.filterwarnings('ignore')


import logging

if os.path.isfile('/app/src/utils/library/bi_online_generation.py'):
	sys.path.append('/app/src/utils/library/')
	print('exist library')
	exist_bi=True
else:
	exist_bi=False

class SBI_Dataset(Dataset):
	def __init__(self,phase='train',image_size=224,n_frames=8):
		
		assert phase in ['train','val','test']
		
		image_list,label_list=init_ff(phase,'frame',n_frames=n_frames)
		
		path_lm='/landmarks/' 
		label_list=[label_list[i] for i in range(len(image_list)) if os.path.isfile(image_list[i].replace('/frames/',path_lm).replace('.png','.npy')) and os.path.isfile(image_list[i].replace('/frames/','/retina/').replace('.png','.npy'))]
		image_list=[image_list[i] for i in range(len(image_list)) if os.path.isfile(image_list[i].replace('/frames/',path_lm).replace('.png','.npy')) and os.path.isfile(image_list[i].replace('/frames/','/retina/').replace('.png','.npy'))]
		self.path_lm=path_lm
		print(f'SBI({phase}): {len(image_list)}')
	

		self.image_list=image_list

		self.image_size=(image_size,image_size)
		self.phase=phase
		self.n_frames=n_frames

		self.transforms=self.get_transforms()
		self.source_transforms = self.get_source_transforms()


	def __len__(self):
		return len(self.image_list)

	def __getitem__(self,idx):
		flag=True
		while flag:
			try:
				filename=self.image_list[idx]
				img=np.array(Image.open(filename))
				landmark=np.load(filename.replace('.png','.npy').replace('/frames/',self.path_lm))[0]
				bbox_lm=np.array([landmark[:,0].min(),landmark[:,1].min(),landmark[:,0].max(),landmark[:,1].max()])
				bboxes=np.load(filename.replace('.png','.npy').replace('/frames/','/retina/'))[:2]
				iou_max=-1
				for i in range(len(bboxes)):
					iou=IoUfrom2bboxes(bbox_lm,bboxes[i].flatten())
					if iou_max<iou:
						bbox=bboxes[i]
						iou_max=iou

				landmark=self.reorder_landmark(landmark)
				if self.phase=='train':
					if np.random.rand()<0.5:
						img,_,landmark,bbox=self.hflip(img,None,landmark,bbox)
						
				img,landmark,bbox,__=crop_face(img,landmark,bbox,margin=True,crop_by_bbox=False)

				img_r,img_f,mask_f=self.self_blending(img.copy(),landmark.copy())

				if self.phase=='train':
					transformed=self.transforms(image=img_f.astype('uint8'),image1=img_r.astype('uint8'))
					img_f=transformed['image']
					img_r=transformed['image1']
					
				
				img_f,_,__,___,y0_new,y1_new,x0_new,x1_new=crop_face(img_f,landmark,bbox,margin=False,crop_by_bbox=True,abs_coord=True,phase=self.phase)
				
				img_r=img_r[y0_new:y1_new,x0_new:x1_new]
				
				img_f=cv2.resize(img_f,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
				img_r=cv2.resize(img_r,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
				

				img_f=img_f.transpose((2,0,1))
				img_r=img_r.transpose((2,0,1))
				flag=False
			except Exception as e:
				print(e)
				idx=torch.randint(low=0,high=len(self),size=(1,)).item()
		
		return img_f,img_r

	
		
	def get_source_transforms(self):
		return alb.Compose([
				alb.Compose([
						alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
						alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=1),
						alb.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=1),
					],p=1),
	
				alb.OneOf([
					RandomDownScale(p=1),
					alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
				],p=1),
				
			], p=1.)

		
	def get_transforms(self):
		return alb.Compose([
			
			alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
			alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
			alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
			alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.5),
			
		], 
		additional_targets={f'image1': 'image'},
		p=1.)


	def randaffine(self,img,mask):
		f=alb.Affine(
				translate_percent={'x':(-0.03,0.03),'y':(-0.015,0.015)},
				scale=[0.95,1/0.95],
				fit_output=False,
				p=1)
			
		g=alb.ElasticTransform(
				alpha=50,
				sigma=7,
				alpha_affine=0,
				p=1,
			)

		transformed=f(image=img,mask=mask)
		img=transformed['image']
		
		mask=transformed['mask']
		transformed=g(image=img,mask=mask)
		mask=transformed['mask']
		return img,mask

		
	def self_blending(self,img,landmark):
		H,W=len(img),len(img[0])
		if np.random.rand()<0.25:
			landmark=landmark[:68]
		if exist_bi:
			logging.disable(logging.FATAL)
			mask=random_get_hull(landmark,img)[:,:,0]
			logging.disable(logging.NOTSET)
		else:
			mask=np.zeros_like(img[:,:,0])
			cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)


		source = img.copy()
		if np.random.rand()<0.5:
			source = self.source_transforms(image=source.astype(np.uint8))['image']
		else:
			img = self.source_transforms(image=img.astype(np.uint8))['image']

		source, mask = self.randaffine(source,mask)

		img_blended,mask=B.dynamic_blend(source,img,mask)
		img_blended = img_blended.astype(np.uint8)
		img = img.astype(np.uint8)

		return img,img_blended,mask
	
	def reorder_landmark(self,landmark):
		landmark_add=np.zeros((13,2))
		for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
			landmark_add[idx]=landmark[idx_l]
		landmark[68:]=landmark_add
		return landmark

	def hflip(self,img,mask=None,landmark=None,bbox=None):
		H,W=img.shape[:2]
		landmark=landmark.copy()
		bbox=bbox.copy()

		if landmark is not None:
			landmark_new=np.zeros_like(landmark)

			
			landmark_new[:17]=landmark[:17][::-1]
			landmark_new[17:27]=landmark[17:27][::-1]

			landmark_new[27:31]=landmark[27:31]
			landmark_new[31:36]=landmark[31:36][::-1]

			landmark_new[36:40]=landmark[42:46][::-1]
			landmark_new[40:42]=landmark[46:48][::-1]

			landmark_new[42:46]=landmark[36:40][::-1]
			landmark_new[46:48]=landmark[40:42][::-1]

			landmark_new[48:55]=landmark[48:55][::-1]
			landmark_new[55:60]=landmark[55:60][::-1]

			landmark_new[60:65]=landmark[60:65][::-1]
			landmark_new[65:68]=landmark[65:68][::-1]
			if len(landmark)==68:
				pass
			elif len(landmark)==81:
				landmark_new[68:81]=landmark[68:81][::-1]
			else:
				raise NotImplementedError
			landmark_new[:,0]=W-landmark_new[:,0]
			
		else:
			landmark_new=None

		if bbox is not None:
			bbox_new=np.zeros_like(bbox)
			bbox_new[0,0]=bbox[1,0]
			bbox_new[1,0]=bbox[0,0]
			bbox_new[:,0]=W-bbox_new[:,0]
			bbox_new[:,1]=bbox[:,1].copy()
			if len(bbox)>2:
				bbox_new[2,0]=W-bbox[3,0]
				bbox_new[2,1]=bbox[3,1]
				bbox_new[3,0]=W-bbox[2,0]
				bbox_new[3,1]=bbox[2,1]
				bbox_new[4,0]=W-bbox[4,0]
				bbox_new[4,1]=bbox[4,1]
				bbox_new[5,0]=W-bbox[6,0]
				bbox_new[5,1]=bbox[6,1]
				bbox_new[6,0]=W-bbox[5,0]
				bbox_new[6,1]=bbox[5,1]
		else:
			bbox_new=None

		if mask is not None:
			mask=mask[:,::-1]
		else:
			mask=None
		img=img[:,::-1].copy()
		return img,mask,landmark_new,bbox_new
	
	def collate_fn(self,batch):
		img_f,img_r=zip(*batch)
		data={}
		data['img']=torch.cat([torch.tensor(img_r).float(),torch.tensor(img_f).float()],0)
		data['label']=torch.tensor([0]*len(img_r)+[1]*len(img_f))
		return data
		

	def worker_init_fn(self,worker_id):                                                          
		np.random.seed(np.random.get_state()[1][0] + worker_id)

if __name__=='__main__':
	import blend as B
	from initialize import *
	from funcs import IoUfrom2bboxes,crop_face,RandomDownScale
	if exist_bi:
		from library.bi_online_generation import random_get_hull
	seed=10
	random.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	image_dataset=SBI_Dataset(phase='test',image_size=256)
	batch_size=64
	dataloader = torch.utils.data.DataLoader(image_dataset,
					batch_size=batch_size,
					shuffle=True,
					collate_fn=image_dataset.collate_fn,
					num_workers=0,
					worker_init_fn=image_dataset.worker_init_fn
					)
	data_iter=iter(dataloader)
	data=next(data_iter)
	img=data['img']
	img=img.view((-1,3,256,256))
	utils.save_image(img, 'loader.png', nrow=batch_size, normalize=False, range=(0, 1))
else:
	from utils import blend as B
	from .initialize import *
	from .funcs import IoUfrom2bboxes,crop_face,RandomDownScale
	if exist_bi:
		from utils.library.bi_online_generation import random_get_hull
#################################################
# C:\Dan_WS\project_src\SelfBlendedImages-master\SelfBlendedImages-master\src\utils\scheduler.py
#################################################

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import _LRScheduler

class LinearDecayLR(_LRScheduler):
    def __init__(self, optimizer, n_epoch, start_decay, last_epoch=-1):
        self.start_decay=start_decay
        self.n_epoch=n_epoch
        super(LinearDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        last_epoch = self.last_epoch
        n_epoch=self.n_epoch
        b_lr=self.base_lrs[0]
        start_decay=self.start_decay
        if last_epoch>start_decay:
            lr=b_lr-b_lr/(n_epoch-start_decay)*(last_epoch-start_decay)
        else:
            lr=b_lr
        return [lr]


if __name__=='__main__':
    model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    optimizer = SGD(model, 0.001)
    s=LinearDecayLR(optimizer, 100, 75)
    ss=[]
    for epoch in range(100):
        optimizer.step()
        s.step()
        ss.append(s.get_lr()[0])

    print(ss)