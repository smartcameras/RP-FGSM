"""
Exploiting Vulnerabilities of Deep Neural Networks for Privacy Protection.
R. Sanchez-Matilla, C.Y. Li, A.S. Shamsabadi, R. Mazzon, A. Cavallaro
IEEE Transactions on Multimedia, July, 2020
If you find this code useful in your research, please consider citing
License in License.txt
"""

import os
import numpy as np
import cv2
from libs.DiffJPEG import DiffJPEG
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair, _quadruple
from torch.autograd import Variable

import glob, os

from PIL import Image
from tqdm import tqdm

import argparse
from scipy import ndimage

from libs.misc_functions import *
from random import shuffle

import time


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {}'.format(device))


class attack():
	"""
		Fast gradient sign untargeted adversarial attack, minimizes the initial class activation
		with iterative grad sign updates
	"""
	def __init__(self, models, epsilon, lb, up, args):
		
		self.models = models
		
		# FGSM parameters
		self.delta = 1/255.
		self.eps = args.eps
		self.numIterations = int(min(self.eps*255*1.25, self.eps*255+4)) * len(self.models)
		
		# Image boundaries
		self.lb = lb
		self.up = up

		# Create the folder to export images if not exists
		self.adv_path = createDirectories(args)

		self.args = args
	
	def getRandom(self, options, howmany):
		
		if howmany !=0:
			div = int(howmany) / len(options)
			rands = np.empty(0, dtype=int)
			for option in options:
				rands = np.concatenate((rands, np.repeat(option, div)), axis=0)

			rest = howmany % len(options)
			if rest != 0:
				rands = np.concatenate((rands, np.random.choice(options, size=rest, p=None, replace=False)), axis=0)
			
			np.random.shuffle(rands)
			unique, counts = np.unique(rands, return_counts=True)
			
			return rands, dict(zip(unique, counts))
		else:
			return 

	def generate(self, original_image, img_name, org_class, classes2target, f1):
		
		all_classes2target = classes2target.copy()

		# Pick randomly target_class from the set
		rand = np.random.randint(0, high=len(classes2target))
		target_class = classes2target[rand]
		target_class_var = Variable(torch.from_numpy(np.asarray([target_class])))

		
		target_class_var = target_class_var.to(device)

		# Process image
		adv_img = preprocess_image(original_image)
		org_img = adv_img.data.clone()

		# Define loss functions
		ce_loss = nn.CrossEntropyLoss()

		text = img_name + '\t'

		# Start iteration
		#randsT = np.random.choice([0,1,2,3], size=self.numIterations, p=None)
		#randsBIT = np.random.choice([1,2,3,4,5,6,7], size=self.numIterations, p=None)
		#randsMEDIAN = np.random.choice([2,3,5], size=self.numIterations, p=None)
		#randsJPEG = np.random.choice([25,50,75], size=self.numIterations, p=None)
		#randsM = np.random.choice(len(self.models), size=self.numIterations, p=None)

		'''
		Randomly choose one of the filter: Bit reduction, Median smoothing or JPEG compression
		Followed by randomly choose one parameter for the chosen filter
		'''
		randsT, counts = self.getRandom([0,1,2,3], self.numIterations)
		if 1 in counts.keys(): 
			randsBIT,_ = self.getRandom([1,2,3,4,5,6,7], counts[1])
			#print('BIT', randsBIT)
		if 2 in counts.keys(): 
			randsMEDIAN,_ = self.getRandom([2,3,5], counts[2])
			#print('MEDIAN', randsMEDIAN)
		if 3 in counts.keys():
			randsJPEG,_ = self.getRandom([25,50,75], counts[3])
			#print('JPG', randsJPEG)
		randsM, _ = self.getRandom(list(np.arange(len(self.models))), self.numIterations)

		for it in range(self.numIterations):
			
			# To [0,1]
			image = standarTo255(adv_img.clone().data)/255
			image = Variable(image, requires_grad=True)
			image = image.to(device)			
			
			if randsT[it]==0:
				imageT = standarise(image.clone())
			
			elif randsT[it]==1:	# Bit reduction
				imageT= standarise(reduce_precision(image.clone(), 2**randsBIT[0], diff=True))
				
			elif randsT[it]==2:	# Median smoothing
				imageT= standarise(medianFilter(image.clone(), int(randsMEDIAN[0])))

			elif randsT[it]==3:	# JPEG compression
				jpeg = DiffJPEG(height=224, width=224, differentiable=True, quality=randsJPEG[0])
				imageT = jpeg(image.clone())					

			image.grad = None
			logit = self.models[randsM[it]](imageT)

			pred_loss = ce_loss(logit, target_class_var)
			pred_loss.backward()

			adv_img.data = addNoise_v2(adv_img.data, image.grad.data, self.delta, self.lb, self.up, clip=True)
			image2save = adv_img.data.clone()
			adv_img.data = from255ToStandar(adv_img.data)

			if randsT[it] == 1:
				randsBIT = np.delete(randsBIT, 0)
			elif randsT[it] == 2:
				randsMEDIAN = np.delete(randsMEDIAN, 0)
			elif randsT[it] == 3:
				randsJPEG = np.delete(randsJPEG, 0)			
	
		# Output of transformed adversarial
		for modelID in range(len(self.models)):
			logit = self.models[modelID](adv_img)
			h_x = F.softmax(logit).data.squeeze()
			probs, idx = h_x.sort(0, True)

			current_class = idx[0]
			current_class_prob = probs[0]
			target_class_prob = h_x[target_class]
			org_class_prob = h_x[org_class[modelID]]	
			text += '{:d}\t{:.5f}\t{:d}\t{:.5f}\t{:d}\t{:.5f}\t'.format(org_class[modelID], org_class_prob.item(), target_class, target_class_prob.item(), current_class.item(), current_class_prob.item())	

		text += '\n'
		f1.write(text)
		saveImage(os.path.join(self.adv_path, img_name), image2save)
		return


if __name__ == '__main__':

	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='./dataset/')
	parser.add_argument('--models', type=str, required=True)
	parser.add_argument('--eps', type=str, required=True)
	parser.add_argument('--gamma', type=float, required=True)
	args = parser.parse_args()
 
	try:
		args.eps = float(args.eps.split('/')[0])/float(args.eps.split('/')[1])	# epsilon defined in [0,1] domain
	except:
		print('Epsilon should be a fraction written like --eps=x/255 (in [0,1] domain)')
  
	args.models = args.models.split(',')
	models, image_list = initialise(args)

	# Log files
	f1, f2, f1_name, f2_name = createLogFiles(args)

	for idx in tqdm(range(1)):
		f1 = open(f1_name, 'a+')
		f2 = open(f2_name, 'a+')

		np.random.seed(0)

		original_image, img_name, lb, ub, org_class, org_class_prob, classes2target = prepareImage(image_list, idx, models, args)

		start = time.time()
		
		RP_FGSM = attack(models, args.eps, lb, ub, args)
		RP_FGSM.generate(original_image, img_name, org_class, classes2target, f1)
		
		end = time.time()
		f2.write('{}\t{:.4f}\n'.format(img_name, end-start))
		f1.close()
		f2.close()


