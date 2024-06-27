from __future__ import print_function
# import time
# import os
import h5py
import numpy as np
# import matplotlib.pyplot as plt
# import scipy.ndimage
import tensorflow as tf

from train_task import train_task
from model import Model


data1_path = 'D:\\others\\project\\U2Fusion-master - testing\\U2Fusion-master - Copy\\train_2.h5'


patch_size = 64
LAM = 0 #80000
LAM_str = '0'
NUM = 30

EPOCHES = [3, 2, 2]
c = [3200, 3500, 100]

def main():
	with tf.Graph().as_default(), tf.Session() as sess:
		model = Model(BATCH_SIZE = 18, INPUT_W = patch_size, INPUT_H = patch_size, is_training = True)
		saver = tf.train.Saver(var_list = model.var_list, max_to_keep = 10)

		tf.summary.scalar('content_Loss', model.content_loss)
		tf.summary.scalar('ssim_Loss', model.ssim_loss)
		tf.summary.scalar('mse_Loss', model.mse_loss)
		tf.summary.scalar('ss1', model.s[0, 0])
		tf.summary.scalar('ss2', model.s[0, 1])
		tf.summary.image('source1', model.SOURCE1, max_outputs = 3)
		tf.summary.image('source2', model.SOURCE2, max_outputs = 3)
		tf.summary.image('fused_result', model.generated_img, max_outputs = 3)
		merged = tf.summary.merge_all()

		'''task1'''
		print('Begin to train the network on task1...\n')
		with tf.device('/cpu:0'):
			source_data1 = h5py.File(data1_path, 'r')
			source_data1 = source_data1['data'][:]
			source_data1 = np.transpose(source_data1, (0, 2, 3, 1))
			print("source_data1 shape:", source_data1.shape)
		writer1 = tf.summary.FileWriter("logs/lam" + LAM_str + "/plot_1", sess.graph)
		train_task(model = model, sess = sess, merged = merged, writer = [writer1], saver = saver, c=c,
		           trainset = source_data1, save_path = './models/lam' + LAM_str + '/task1/', lam = LAM, task_ind = 1,
		           EPOCHES = EPOCHES[0])

if __name__ == '__main__':
	main()
