#coding:utf-8
__author__ = 'ASUS'
import cv2 as cv
import os
import glob
import numpy as np
import cPickle
import string
import random

def subSampleImg(image,win_w,newShape):
	# 输入图像，滑动窗宽度，输出图片尺寸

	# newShape = (newShape[0],int(newShape[1]/1.75),newShape[2])

	win_h = int((1.0*image.shape[0]/image.shape[1])*win_w)  #滑动窗的宽高比例与输入图像一致
	nstep_w = image.shape[1]-win_w
	nstep_h = image.shape[0]-win_h
	# imgResize = np.empty(image.shape,'uint8')
	if 3 == newShape[2]:
		isColor = True
	else: isColor = False

	if 0 == nstep_w:
		subSamples = np.empty((1,newShape[2],newShape[0],newShape[1]),'uint8')
		imgResize = cv.resize(image,(newShape[1],newShape[0]))  ######opencv的resize里的参数是宽、高


		if not isColor:
			# halfImg = imgResize[:,-newShape[1]-1:-1]
			subSamples[0,0] = imgResize
		else:
			# halfImg = imgResize[:,-newShape[1]-1:-1]
			img_temp = imgResize.swapaxes(0,2).swapaxes(1,2)
			subSamples[0] = img_temp
		#
		# cv.imshow("halfImg",halfImg)
		# cv.waitKey(0)

		return subSamples

	else:
		subSamples = np.empty((nstep_w*nstep_h,newShape[2],newShape[0],newShape[1]),'uint8')   #n*34*34*3
		for i in range(nstep_h):
			for j in range(nstep_w):
				subImg = image[i:i+win_h,j:j+win_w]
				imgResize = cv.resize(subImg,(newShape[1],newShape[0]))#######

				if not isColor:
					# halfImg = imgResize[:,-newShape[1]-1:-1]
					subSamples[i*nstep_w+j,0,:,:] = imgResize
				else:
					# halfImg = imgResize[:,-newShape[1]-1:-1]
					img_temp = imgResize.swapaxes(0,2).swapaxes(1,2)
					subSamples[i*nstep_w+j] = img_temp

		#返回中间张
		subSamples_re = np.empty((2,newShape[2],newShape[0],newShape[1]),'uint8')   #n*34*34*3
		subSamples_re[0] = subSamples[nstep_w*nstep_h/3]
		subSamples_re[1] = subSamples[-nstep_w*nstep_h/3]

		# testImg1 = subSamples_re[0,0]
		# cv.imshow("1",testImg1)
		# testImg1 = subSamples_re[0,0]
		# cv.imshow("2",testImg1)
		# cv.waitKey(500)


		return subSamples_re




if __name__ == '__main__':
	all_dir = 'H:\\veheicleSample\\headOfCar\\2-3.5\\1detailClass\\*'
	out_dir = 'H:\\veheicleSample\\headOfCar\\samplePKL\\detailClass'

	resize_row = 98
	resize_col = 174
	resize_nchan = 1
	resize_shape = (resize_row,resize_row,resize_nchan)
	nplus = 7

	n_Class = len(glob.glob(all_dir))

	# 数据集是否为空
	train_empty = True
	valid_empty = True
	test_empty = True

	train_label = []
	valid_label = []
	test_label = []

	# print nClass
	for detailDir in glob.glob(all_dir):
		num_nClass = len(glob.glob(detailDir+'\*.bmp'))
		# print num_nClass
		n_th = 0
		n_train = 0
		n_valid = 0
		n_test = 0
		dir_name = os.path.split(detailDir)[1]
		class_label = string.atoi(dir_name)
		if 50 == class_label:  #只取前n类
			break

		for file in (glob.glob(detailDir+'\*.bmp')):
			threshold1 = (3.0/5)*num_nClass+0.01
			threshold2 = (4.0/5)*num_nClass+0.01
			# 判断数据类型，并记录数量
			if n_th<threshold1:
				dataType = 'train'
				n_train = n_train+1

			elif n_th>threshold1 and n_th<threshold2:
				dataType = 'valid'
				n_valid = n_valid + 1

			else:
				dataType = 'test'
				n_test = n_test + 1
			n_th = n_th + 1

			# 图像缩放与切割
			srcImg = cv.imread(file,0)
			resizeImg = cv.resize(srcImg,(resize_col,resize_row))  #注意，是尺寸，宽高
			halfImg = resizeImg[: ,:resize_row]
			halfImg = cv.equalizeHist(halfImg)

			# print "row=%d  col=%d" % (halfImg.shape[0],halfImg.shape[1])
			# cv.imshow("halfImg",halfImg)
			# cv.waitKey(0)


			#存入数据
			if dataType == 'train':
				for ii in range(2):
					if 1 == ii:
						# halfImg = cv.flip(halfImg,1)
						halfImg = resizeImg[: ,-resize_row:]
						halfImg = cv.flip(halfImg,1)
						halfImg = cv.equalizeHist(halfImg)
					# cv.imshow("flip",halfImg)
					# cv.waitKey(0)
					# cv.imshow("src",halfImg)

					# cv.imshow("ehist",halfImg)
					# cv.waitKey(0)

					subImg1 = subSampleImg(halfImg,halfImg.shape[1]-0,resize_shape)
					subImg2 = subSampleImg(halfImg,halfImg.shape[1]-2,resize_shape)
					subImg3 = subSampleImg(halfImg,halfImg.shape[1]-4,resize_shape)
					subImg4 = subSampleImg(halfImg,halfImg.shape[1]-6,resize_shape)

					if train_empty :
						train_data = subImg1.reshape((subImg1.shape[0],np.prod(subImg1.shape[1:])))
						train_empty = False
					else:
						train_data = np.row_stack((train_data,subImg1.reshape((subImg1.shape[0],np.prod(subImg1.shape[1:])))))

					train_data = np.row_stack((train_data,subImg2.reshape((subImg2.shape[0],np.prod(subImg2.shape[1:])))))
					train_data = np.row_stack((train_data,subImg3.reshape((subImg3.shape[0],np.prod(subImg3.shape[1:])))))
					train_data = np.row_stack((train_data,subImg4.reshape((subImg4.shape[0],np.prod(subImg4.shape[1:])))))

					# 添加标签
					for i in range(nplus):
						train_label.append(class_label)

			if dataType == 'valid':
				subImg1 = subSampleImg(halfImg,halfImg.shape[1]-0,resize_shape)
				if valid_empty :
					valid_data = subImg1.reshape((subImg1.shape[0],np.prod(subImg1.shape[1:])))
					valid_empty = False
				else:
					valid_data = np.row_stack((valid_data,subImg1.reshape((subImg1.shape[0],np.prod(subImg1.shape[1:])))))

				# 添加标签
				valid_label.append(class_label)

			if dataType == 'test':
				subImg1 = subSampleImg(halfImg,halfImg.shape[1]-0,resize_shape)
				if test_empty :
					test_data = subImg1.reshape((subImg1.shape[0],np.prod(subImg1.shape[1:])))
					test_empty = False
				else:
					test_data = np.row_stack((test_data,subImg1.reshape((subImg1.shape[0],np.prod(subImg1.shape[1:])))))

				# 添加标签
				test_label.append(class_label)



	#读取数据完毕，保存数据

	#由list转为数列
	train_label = np.array(train_label)
	valid_label = np.array(valid_label)
	test_label = np.array(test_label)

	# saveFile = open(out_dir+'/' + 'all_half_gray.pkl','wb')
	# cPickle.dump([(train_data,train_label),(valid_data,valid_label),\
	# 			  (test_data,test_label) ],saveFile)
	# saveFile.close()

	trainNum = train_data.shape[0]
	randomIndex = range(0,trainNum)
	random.shuffle(randomIndex)

	train_data_r = np.empty(train_data.shape,'uint8')
	train_label_r = np.empty(train_label.shape,'uint8')

	for i in range(trainNum):
		train_data_r[i] = train_data[randomIndex[i]]
		train_label_r[i] = train_label[randomIndex[i]]

	# # scale = 34.0/49.0
	# testData_x = trainData_x[trainData_x.shape[0] *scale:]
	# testData_y = trainData_y[trainData_y.shape[0] *scale:]
	# trainData_x = trainData_x[:trainData_x.shape[0] *scale]
	# trainData_y = trainData_y[:trainData_y.shape[0] *scale]

	saveFile = open(out_dir+'/' + 'HalfGrayFlipEqhist_random50.pkl','wb')
	cPickle.dump([(train_data_r,train_label_r),(valid_data,valid_label),\
				  (test_data,test_label) ],saveFile)
	saveFile.close()

