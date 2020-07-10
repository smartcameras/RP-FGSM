"""
Exploiting Vulnerabilities of Deep Neural Networks for Privacy Protection.
R. Sanchez-Matilla, C.Y. Li, A.S. Shamsabadi, R. Mazzon, A. Cavallaro
IEEE Transactions on Multimedia, July, 2020
If you find this code useful in your research, please consider citing
License in License.txt
"""

import copy
import cv2
import numpy as np

import torch
from torch.autograd import Variable

from torchvision import models
from torch.nn import functional as F

from PIL import Image

import csv
import os
import glob

from functools import reduce

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def addNoise(img, grads, delta, lb, ub, clip=True):

	adv_noise = delta * sum(grads).sign()
	
	# De-standarise image
	img = destandarise(img)
	
	# Add noise
	img -= adv_noise

	# Clip
	if clip:
		img = torch.min(torch.max(img, lb), ub)
	else:
		img = torch.clamp(x, min=0, max=1)

	# Make sure image goes to 255 for accounting with rounding
	return standarTo255(standarise(img))

def addNoise_v2(img, grad, delta, lb, ub, clip=True):

	adv_noise = delta * grad.sign()
	
	# De-standarise image
	img = destandarise(img)
	#import pdb; pdb.set_trace()
	# Add noise
	img -= adv_noise

	# Clip
	if clip:
		img = torch.min(torch.max(img, lb), ub)
	else:
		img = torch.clamp(x, min=0, max=1)

	# Make sure image goes to 255 for accounting with rounding
	return standarTo255(standarise(img))

def calculate_bounds(img, epsilon):
   
	img = np.float32(img)
	img = np.ascontiguousarray(img[..., ::-1])
	img = img.transpose(2, 0, 1)  # Convert array to D,W,H
	
	# Normalize the channels
	for c, _ in enumerate(img):
		img[c] /= 255

	img = torch.from_numpy(img).float()
	img.unsqueeze_(0)

	lb = Variable(img.clone(), requires_grad=False)
	ub = Variable(img.clone(), requires_grad=False)
 
	# Compute the bounds
	lb -= epsilon
	ub += epsilon

	# Clip between [0,1]
	lb = torch.clamp(lb, 0., 1.)
	ub = torch.clamp(ub, 0., 1.)
	
	lb = lb.to(device)
	ub = ub.to(device)

	#cv2.imwrite('./lb.png', recreate_image(lb))
	#cv2.imwrite('./ub.png', recreate_image(ub))
	return lb, ub


def saveImage(path, img):
	img = img.squeeze().cpu().detach().numpy()
	img = np.uint8(img).transpose(1, 2, 0)
	img = img[..., ::-1]
	cv2.imwrite(path, img)

def preprocess_image(cv2im, resize_im=True):

	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	
	# Resize image
	if resize_im:
		cv2im = cv2.resize(cv2im, (224, 224), interpolation=cv2.INTER_LINEAR)
	im_as_arr = np.float32(cv2im)
	im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
	im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
	# Normalize the channels
	for channel, _ in enumerate(im_as_arr):
		im_as_arr[channel] /= 255
		im_as_arr[channel] -= mean[channel]
		im_as_arr[channel] /= std[channel]
	# Convert to float tensor
	im_as_ten = torch.from_numpy(im_as_arr).float()
	# Add one more channel to the beginning. Tensor shape = 1,3,224,224
	im_as_ten.unsqueeze_(0)
	# Convert to Pytorch variable
	im_as_var = Variable(im_as_ten.to(device), requires_grad=True)
	return im_as_var
	

def standarTo255(img):
		
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	
	for c in range(3):
		img[0,c,:,:] *= std[c]
		img[0,c,:,:] += mean[c]
	
	img[img > 1] = 1
	img[img < 0] = 0

	return (img*255).round()
	


def from255ToStandar(img):

	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]

	for c in range(3):
		img[0,c,:,:] /= 255
		img[0,c,:,:] -= mean[c]
		img[0,c,:,:] /= std[c]

	return img.float()

def recreate_image(im_as_var):

	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	recreated_im = copy.copy(im_as_var.cpu().data.numpy()[0])
	for c in range(3):
		recreated_im[c] *= std[c]
		recreated_im[c] += mean[c]
	recreated_im[recreated_im > 1] = 1
	recreated_im[recreated_im < 0] = 0
	recreated_im = np.round(recreated_im * 255)
	recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
	# Convert RBG to GBR
	recreated_im = recreated_im[..., ::-1]
	return recreated_im


def destandarise(img):
	
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	for c in range(3):
		img[0,c,:,:] *= std[c]
		img[0,c,:,:] += mean[c]
	return img


def standarise(img):
	
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	for c in range(3):
		img[0,c,:,:] -= mean[c]
		img[0,c,:,:] /= std[c]
	 
	return img

def initialise(args):

	image_list = glob.glob('{}*.png'.format(args.dataset)) + glob.glob('{}*.jpg'.format(args.dataset))

	allModels = []
	for arch in args.models:
		model_file = 'models/%s_places365.pth.tar' % arch
		if not os.access(model_file, os.W_OK):

			if not os.path.exists('models'):
				os.makedirs('models')

			weight_url = 'http://places2.csail.mit.edu/models_places365/' + arch + '_places365.pth.tar'
			print(weight_url)
			os.system('wget ' + weight_url)
			os.system('mv *.pth.tar ./models')
			print('Model downloaded!')

		model = models.__dict__[arch](num_classes=365)
		checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
		state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
		model.load_state_dict(state_dict)
		model.to(device)
		model.eval()
		allModels.append(model)

	return allModels, image_list



def prepareImage(image_list, index, models, args):

	# Read image
	img_path = image_list[index]
	img_name = img_path.split('/')[-1]
	original_image = cv2.imread(img_path, 1)
	original_image = cv2.resize(original_image, (224, 224), interpolation=cv2.INTER_LINEAR)

	# Image boundaries
	lb, ub = calculate_bounds(original_image, args.eps)
	
	# Process image
	image = preprocess_image(original_image)

	org_class = []
	org_class_prob = []
	list_classes2target = []
	for model in models:
		
		# forward pass
		logit = model.forward(image)
		h_x = F.softmax(logit).data.squeeze()
		probs, idx = h_x.sort(0, True)

		probs = np.array(probs.cpu())
		idx = np.array(idx.cpu())
	   
		org_class.append(idx[0].item())
		org_class_prob.append(probs[0].item())

		cumSum = np.cumsum(probs)
		classes2target = idx[cumSum > args.gamma]
		classes2target = classes2target[1:]
		list_classes2target.append(classes2target)

	#Intersection
	classes2target = reduce(np.intersect1d, (list_classes2target)) #functools

	return original_image, img_name, lb, ub, org_class, org_class_prob, classes2target


def createLogFiles(args):
	if not os.path.exists('results'):
		os.makedirs('results')
	f1_name = 'results/log'
	f2_name = 'results/logTimes'
	for model in args.models:
		f1_name += '_{}'.format(model)
		f2_name += '_{}'.format(model)
	f1_name += '_eps{:.5f}_gamma{:.2f}.txt'.format(args.eps,args.gamma)
	f2_name += '_eps{:.5f}_gamma{:.2f}.txt'.format(args.eps,args.gamma)
	f1 = open(f1_name,"w")
	f2 = open(f2_name,"w")
	return f1, f2, f1_name, f2_name


def createDirectories(args):
	adv_path = 'results/adv'
	for model in args.models:
		adv_path += '_{}'.format(model)
	adv_path += '_eps{:.5f}_gamma{:.2f}'.format(args.eps,args.gamma)
   
	if not os.path.exists(adv_path):
		os.makedirs(adv_path)

	return adv_path


def padding(img, kernel):
	ih, iw = img.size()[2:]
	if ih % 1 == 0:
		ph = max(kernel - 1, 0)
	else:
		ph = max(kernel - (ih % 1), 0)
	if iw % 1 == 0:
		pw = max(kernel - 1, 0)
	else:
		pw = max(kernel - (iw % 1), 0)
	pl = pw // 2
	pr = pw - pl
	pt = ph // 2
	pb = ph - pt
	padding = (pl, pr, pt, pb)
	return padding

def medianFilter(img, kernel):
	img = F.pad(img, padding(img, kernel), mode='reflect')
	img = img.unfold(2, kernel, 1).unfold(3, kernel, 1)
	img = img.contiguous().view(img.size()[:4] + (-1,)).median(dim=-1)[0]
	return img


def diff_round(x):
	return torch.round(x) + (x - torch.round(x))**3

def reduce_precision(x, npp, diff=False):
	npp_int = npp - 1
	if diff==False:
		x_int = (x * npp_int).round()
	else:
		x_int = diff_round(x * npp_int)
	x_float = x_int / npp_int
	return x_float


def jpeg_compression(x, quality, args):
	nx = x.squeeze().cpu().detach().numpy()
	nx = np.uint8(nx*255).transpose(1, 2, 0)
	nx = nx[..., ::-1]

	tmpImageName = 'tmp_'
	for model in args.models:
		tmpImageName += model
	tmpImageName += '.jpg'

	cv2.imwrite(tmpImageName, nx, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
	img_jpeg = preprocess_image(cv2.imread(tmpImageName))
	img_jpeg = standarTo255(img_jpeg.clone().data)/255
	return img_jpeg

