#################################################
# C:\Dan_WS\project_src\SelfBlendedImages-master\SelfBlendedImages-master\src\inference\datasets.py
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


def init_ff(dataset='all',phase='test'):
	assert dataset in ['all','Deepfakes','Face2Face','FaceSwap','NeuralTextures']
	original_path='data/FaceForensics++/original_sequences/youtube/raw/videos/'
	folder_list = sorted(glob(original_path+'*'))

	list_dict = json.load(open(f'data/FaceForensics++/{phase}.json','r'))
	filelist=[]
	for i in list_dict:
		filelist+=i
	image_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]
	label_list=[0]*len(image_list)


	if dataset=='all':
		fakes=['Deepfakes','Face2Face','FaceSwap','NeuralTextures']
	else:
		fakes=[dataset]

	folder_list=[]
	for fake in fakes:
		fake_path=f'data/FaceForensics++/manipulated_sequences/{fake}/raw/videos/'
		folder_list_all=sorted(glob(fake_path+'*'))
		folder_list+=[i for i in folder_list_all if os.path.basename(i)[:3] in filelist]
	label_list+=[1]*len(folder_list)
	image_list+=folder_list
	return image_list,label_list



def init_dfd():
	real_path='data/FaceForensics++/original_sequences/actors/raw/videos/*.mp4'
	real_videos=sorted(glob(real_path))
	fake_path='data/FaceForensics++/manipulated_sequences/DeepFakeDetection/raw/videos/*.mp4'
	fake_videos=sorted(glob(fake_path))

	label_list=[0]*len(real_videos)+[1]*len(fake_videos)

	image_list=real_videos+fake_videos

	return image_list,label_list


def init_dfdc():
		
	label=pd.read_csv('data/DFDC/labels.csv',delimiter=',')
	folder_list=[f'data/DFDC/videos/{i}' for i in label['filename'].tolist()]
	label_list=label['label'].tolist()
	
	return folder_list,label_list


def init_dfdcp(phase='test'):

	phase_integrated={'train':'train','val':'train','test':'test'}

	all_img_list=[]
	all_label_list=[]

	with open('data/DFDCP/dataset.json') as f:
		df=json.load(f)
	fol_lab_list_all=[[f"data/DFDCP/{k.split('/')[0]}/videos/{k.split('/')[-1]}",df[k]['label']=='fake'] for k in df if df[k]['set']==phase_integrated[phase]]
	name2lab={os.path.basename(fol_lab_list_all[i][0]):fol_lab_list_all[i][1] for i in range(len(fol_lab_list_all))}
	fol_list_all=[f[0] for f in fol_lab_list_all]
	fol_list_all=[os.path.basename(p)for p in fol_list_all]
	folder_list=glob('data/DFDCP/method_*/videos/*/*/*.mp4')+glob('data/DFDCP/original_videos/videos/*/*.mp4')
	folder_list=[p for p in folder_list if os.path.basename(p) in fol_list_all]
	label_list=[name2lab[os.path.basename(p)] for p in folder_list]
	

	return folder_list,label_list




def init_ffiw():
	# assert dataset in ['real','fake']
	path='data/FFIW/FFIW10K-v1-release/'
	folder_list=sorted(glob(path+'source/val/videos/*.mp4'))+sorted(glob(path+'target/val/videos/*.mp4'))
	label_list=[0]*250+[1]*250
	return folder_list,label_list



def init_cdf():

	image_list=[]
	label_list=[]

	video_list_txt='data/Celeb-DF-v2/List_of_testing_videos.txt'
	with open(video_list_txt) as f:
		
		folder_list=[]
		for data in f:
			# print(data)
			line=data.split()
			# print(line)
			path=line[1].split('/')
			folder_list+=['data/Celeb-DF-v2/'+path[0]+'/videos/'+path[1]]
			label_list+=[1-int(line[0])]
		return folder_list,label_list
		
#################################################
# C:\Dan_WS\project_src\SelfBlendedImages-master\SelfBlendedImages-master\src\inference\inference_dataset.py
#################################################

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms,models,utils
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
import sys
import random
import shutil
from model import Detector
import argparse
from datetime import datetime
from tqdm import tqdm
from retinaface.pre_trained_models import get_model
from preprocess import extract_frames
from datasets import *
from sklearn.metrics import confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def main(args):

    model=Detector()
    model=model.to(device)
    cnn_sd=torch.load(args.weight_name)["model"]
    model.load_state_dict(cnn_sd)
    model.eval()

    face_detector = get_model("resnet50_2020-07-20", max_size=2048,device=device)
    face_detector.eval()

    if args.dataset == 'FFIW':
        video_list,target_list=init_ffiw()
    elif args.dataset == 'FF':
        video_list,target_list=init_ff()
    elif args.dataset == 'DFD':
        video_list,target_list=init_dfd()
    elif args.dataset == 'DFDC':
        video_list,target_list=init_dfdc()
    elif args.dataset == 'DFDCP':
        video_list,target_list=init_dfdcp()
    elif args.dataset == 'CDF':
        video_list,target_list=init_cdf()
    else:
        NotImplementedError

    output_list=[]
    for filename in tqdm(video_list):
        try:
            face_list,idx_list=extract_frames(filename,args.n_frames,face_detector)

            with torch.no_grad():
                img=torch.tensor(face_list).to(device).float()/255
                pred=model(img).softmax(1)[:,1]
                
                
            pred_list=[]
            idx_img=-1
            for i in range(len(pred)):
                if idx_list[i]!=idx_img:
                    pred_list.append([])
                    idx_img=idx_list[i]
                pred_list[-1].append(pred[i].item())
            pred_res=np.zeros(len(pred_list))
            for i in range(len(pred_res)):
                pred_res[i]=max(pred_list[i])
            pred=pred_res.mean()
        except Exception as e:
            print(e)
            pred=0.5
        output_list.append(pred)

    auc=roc_auc_score(target_list,output_list)
    print(f'{args.dataset}| AUC: {auc:.4f}')






if __name__=='__main__':

    seed=1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    parser=argparse.ArgumentParser()
    parser.add_argument('-w',dest='weight_name',type=str)
    parser.add_argument('-d',dest='dataset',type=str)
    parser.add_argument('-n',dest='n_frames',default=32,type=int)
    args=parser.parse_args()

    main(args)



#################################################
# C:\Dan_WS\project_src\SelfBlendedImages-master\SelfBlendedImages-master\src\inference\inference_image.py
#################################################

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets,transforms,models,utils
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
import sys
import random
import shutil
from model import Detector
import argparse
from datetime import datetime
from tqdm import tqdm
from retinaface.pre_trained_models import get_model
from preprocess import extract_face
import warnings
warnings.filterwarnings('ignore')

def main(args):

    model=Detector()
    model=model.to(device)
    cnn_sd=torch.load(args.weight_name)["model"]
    model.load_state_dict(cnn_sd)
    model.eval()

    frame = cv2.imread(args.input_image)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_detector = get_model("resnet50_2020-07-20", max_size=max(frame.shape),device=device)
    face_detector.eval()

    face_list=extract_face(frame,face_detector)

    with torch.no_grad():
        img=torch.tensor(face_list).to(device).float()/255
        # torchvision.utils.save_image(img, f'test.png', nrow=8, normalize=False, range=(0, 1))
        pred=model(img).softmax(1)[:,1].cpu().data.numpy().tolist()

    print(f'fakeness: {max(pred):.4f}')


if __name__=='__main__':

    seed=1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    parser=argparse.ArgumentParser()
    parser.add_argument('-w',dest='weight_name',type=str)
    parser.add_argument('-i',dest='input_image',type=str)
    args=parser.parse_args()

    main(args)




#################################################
# C:\Dan_WS\project_src\SelfBlendedImages-master\SelfBlendedImages-master\src\inference\inference_video.py
#################################################


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms,models,utils
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
import sys
import random
import shutil
from model import Detector
import argparse
from datetime import datetime
from tqdm import tqdm
from retinaface.pre_trained_models import get_model
from preprocess import extract_frames
import warnings
warnings.filterwarnings('ignore')

def main(args):

    model=Detector()
    model=model.to(device)
    cnn_sd=torch.load(args.weight_name)["model"]
    model.load_state_dict(cnn_sd)
    model.eval()

    face_detector = get_model("resnet50_2020-07-20", max_size=2048,device=device)
    face_detector.eval()

    face_list,idx_list=extract_frames(args.input_video,args.n_frames,face_detector)

    with torch.no_grad():
        img=torch.tensor(face_list).to(device).float()/255
        pred=model(img).softmax(1)[:,1]
        
        
    pred_list=[]
    idx_img=-1
    for i in range(len(pred)):
        if idx_list[i]!=idx_img:
            pred_list.append([])
            idx_img=idx_list[i]
        pred_list[-1].append(pred[i].item())
    pred_res=np.zeros(len(pred_list))
    for i in range(len(pred_res)):
        pred_res[i]=max(pred_list[i])
    pred=pred_res.mean()

    print(f'fakeness: {pred:.4f}')






if __name__=='__main__':

    seed=1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    parser=argparse.ArgumentParser()
    parser.add_argument('-w',dest='weight_name',type=str)
    parser.add_argument('-i',dest='input_video',type=str)
    parser.add_argument('-n',dest='n_frames',default=32,type=int)
    args=parser.parse_args()

    main(args)



#################################################
# C:\Dan_WS\project_src\SelfBlendedImages-master\SelfBlendedImages-master\src\inference\model.py
#################################################
import torch
from torch import nn
import torchvision
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet




class Detector(nn.Module):

    def __init__(self):
        super(Detector, self).__init__()
        self.net=EfficientNet.from_pretrained("efficientnet-b4",advprop=True,num_classes=2)
        

    def forward(self,x):
        x=self.net(x)
        return x
    
    



#################################################
# C:\Dan_WS\project_src\SelfBlendedImages-master\SelfBlendedImages-master\src\inference\preprocess.py
#################################################



import numpy as np
import cv2
from PIL import Image
import sys
from tqdm import tqdm

def extract_frames(filename,num_frames,model,image_size=(380,380)):
	cap_org = cv2.VideoCapture(filename)
	
	if not cap_org.isOpened():
		print(f'Cannot open: {filename}')
		# sys.exit()
		return []
	
	croppedfaces=[]
	idx_list=[]
	frame_count_org = int(cap_org.get(cv2.CAP_PROP_FRAME_COUNT))
	
	frame_idxs = np.linspace(0, frame_count_org - 1, num_frames, endpoint=True, dtype=int)
	for cnt_frame in range(frame_count_org): 
		ret_org, frame_org = cap_org.read()
		height,width=frame_org.shape[:-1]
		if not ret_org:
			tqdm.write('Frame read {} Error! : {}'.format(cnt_frame,os.path.basename(filename)))
			break
		
		if cnt_frame not in frame_idxs:
			continue

		frame = cv2.cvtColor(frame_org, cv2.COLOR_BGR2RGB)
		
		faces = model.predict_jsons(frame)
		try:
			if len(faces)==0:
				tqdm.write('No faces in {}:{}'.format(cnt_frame,os.path.basename(filename)))
				continue

			size_list=[]
			croppedfaces_temp=[]
			idx_list_temp=[]
			
			for face_idx in range(len(faces)):
				x0,y0,x1,y1=faces[face_idx]['bbox']
				bbox=np.array([[x0,y0],[x1,y1]])
				croppedfaces_temp.append(cv2.resize(crop_face(frame,None,bbox,False,crop_by_bbox=True,only_img=True,phase='test'),dsize=image_size).transpose((2,0,1)))
				idx_list_temp.append(cnt_frame)
				size_list.append((x1-x0)*(y1-y0))
			
			max_size=max(size_list)
			croppedfaces_temp=[f for face_idx,f in enumerate(croppedfaces_temp) if size_list[face_idx]>=max_size/2]
			idx_list_temp=[f for face_idx,f in enumerate(idx_list_temp) if size_list[face_idx]>=max_size/2]
			croppedfaces+=croppedfaces_temp
			idx_list+=idx_list_temp	
		except Exception as e:
			print(f'error in {cnt_frame}:{filename}')
			print(e)
			continue
	cap_org.release()

	

	return croppedfaces,idx_list

def extract_face(frame,model,image_size=(380,380)):
	
	
	faces = model.predict_jsons(frame)

	if len(faces)==0:
		print('No face is detected' )
		return []

	croppedfaces=[]
	for face_idx in range(len(faces)):
		x0,y0,x1,y1=faces[face_idx]['bbox']
		bbox=np.array([[x0,y0],[x1,y1]])
		croppedfaces.append(cv2.resize(crop_face(frame,None,bbox,False,crop_by_bbox=True,only_img=True,phase='test'),dsize=image_size).transpose((2,0,1)))
	
	return croppedfaces


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
