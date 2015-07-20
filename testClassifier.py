#coding: gb18030
#coding:utf8#
# -*-coding:utf8-*-#


__author__ = 'Administrator'


import os
import sys
import time
import numpy
from PIL import Image


import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

import PIL
import cv2 as cv
import os
import glob
import numpy as np
import cPickle
import string
import random


import trainDetailCNN_half_gray as trainCNN

def evaluate_carModel(nkerns=[10,20],usePreParams=True):

	threshold = 0.8

	dir_in = 'H:\\veheicleSample\\headOfCar\\2-3.5\\1detailClass\\*'
	searchPath = 'H:\\veheicleSample\\headOfCar\\2-3.5\\1detailClass\\'

	paramsName = '50class_params.pkl'
	paramsPath =  paramsName

	mean_dir = 'H:\\veheicleSample\\headOfCar\\samplePKL\\detailClass'

	Nclass = 90
	# 设置参数
	#输入图像尺寸
	resize_row = 98
	resize_col = 174


	imgNChannels = 1
	imgRows = 98
	imgCols = 98

	L0_imgRows = imgRows
	L0_imgCols = imgCols

	filterSize = [11,7]  # 卷积核宽度
	L0PoolSize = (4,4)
	L1PoolSize = (2,2)
	HL_nout = 500  #隐层输入
	classNum = 50  #分类数量

	# f = open("ZCA.pkl",'rb')
	# ZCAWhite = cPickle.load(f)
	# f.close()

	x = T.matrix('x')

	print '...building the model'

	layer0_params,layer1_params,layer2_params,layer3_params=trainCNN.load_params(paramsPath)

	rng = numpy.random.RandomState(23455)
	L0_imgShape = (1,imgNChannels,L0_imgRows,L0_imgCols)
	layer0_input = x.reshape(L0_imgShape)
	layer0 = trainCNN.LeNetConvPoolLayer(
		rng,
		input = layer0_input,
        params_W=layer0_params[0],
        params_b=layer0_params[1],
		usePreParams=usePreParams,
		image_shape = L0_imgShape,
		filter_shape = (nkerns[0],imgNChannels,filterSize[0],filterSize[0]),
		poolsize=L0PoolSize
	)


	# 第二个卷积+maxpool层,输入是上层的输出，即(batch_size, nkerns[0], 26, 21)
	L1_imgShapeRows = (L0_imgRows - filterSize[0]+1)/L0PoolSize[0]
	L1_imgShapeCols = (L0_imgCols - filterSize[0]+1)/L0PoolSize[0]
	layer1 = trainCNN.LeNetConvPoolLayer(
		rng,
		input=layer0.output,
		params_W=layer1_params[0],
        params_b=layer1_params[1],
		usePreParams=usePreParams,
		image_shape=(1, nkerns[0],L1_imgShapeRows,L1_imgShapeCols),
		filter_shape=(nkerns[1], nkerns[0], filterSize[1], filterSize[1]),
		poolsize=L1PoolSize
	)

	layer2_input = layer1.output.flatten(2)
	HL_nout = HL_nout
	L2_imgShapeRows = (L1_imgShapeRows - filterSize[1] + 1)/L1PoolSize[0]
	L2_imgShapeCols = (L1_imgShapeCols - filterSize[1] + 1)/L1PoolSize[0]
	layer2 = trainCNN.HiddenLayer(
        rng,
        input=layer2_input,
        params_W=layer2_params[0],
        params_b=layer2_params[1],
		usePreParams=usePreParams,
		n_in=nkerns[1] * L2_imgShapeRows * L2_imgShapeCols,
		# n_in=nkerns[1] * (layer1.output.shape[-1]**2),
        n_out=HL_nout,      #全连接层输出神经元的个数，自己定义的，可以根据需要调节
        activation=T.tanh
    )


	# ss =  (layer2.output).get_value()
	# layer3_input = layer2.output.reshape(1,100)
	layer3_input = layer2.output.flatten(2)   #(nkerns[2]),
	layer3 = trainCNN.LogisticRegression(
		input=layer3_input,
		params_W=layer3_params[0],
        params_b=layer3_params[1],
		usePreParams=usePreParams,
		n_in=HL_nout, n_out=classNum)

	carRecog = theano.function(
		[x],
		layer3.y_pred,
	)


	###############
	#测试
	###############

	for dir_detail in glob.glob( dir_in ):
		for img_path in glob.glob(dir_detail+'\\*.bmp' ):
			# print dir_detail
			# print img_path
			inputImg = cv.imread(img_path,0)
			ResizeImg_temp = cv.resize(inputImg,(resize_col,resize_row))
			ResizeImg = ResizeImg_temp[: ,:resize_row]
			ResizeImg = cv.equalizeHist(ResizeImg)
			read_file = open(mean_dir + '\\'+'x_mean.pkl','rb')
			x_mean = cPickle.load(read_file)
			read_file.close()

			testData_x =  np.asarray(ResizeImg,dtype='float32').flatten(0)/256.0
			testData_x = (testData_x).reshape((1,testData_x.shape[0]))

			objectID = int(carRecog(testData_x))

			num_dir_detail = int(dir_detail.split('\\')[-1])
			if not objectID == num_dir_detail:
				if objectID < 10:
					out_path = '000%d' % objectID
				else:
					out_path = '00%d' % objectID

				out_path = dir_in[:-1] + out_path
				for out_img_path in glob.glob(out_path+'\\*.bmp' ):
					out_path = out_img_path
					break

				print 'src_class = %d, dst_class = %d' %(num_dir_detail,objectID)
				out_img = cv.imread(out_path,0)
				cv.imshow('src',inputImg)
				cv.imshow('out',out_img)
				cv.waitKey(0)
				cv.destroyAllWindows()







	inputImg = cv.imread(inputImgPath,0)
	ResizeImg_temp = cv.resize(inputImg,(imgCols,imgRows))
	ResizeImg = ResizeImg_temp[:,int(-ResizeImg_temp.shape[1]/1.75-1):-1]
	read_file = open('x_mean.pkl','rb')
	x_mean = cPickle.load(read_file)
	read_file.close()

	testData_x =  np.asarray(ResizeImg,dtype='float32').flatten(0)/256.0
	testData_x = (testData_x - x_mean).reshape((1,testData_x.shape[0]))

	objectID = carRecog(testData_x)


	correctNum = 0
	errorNum = 0
	p = 0
	n = 0
	i = 0
	for file in glob.glob(searchPath+'/*.bmp'):
		filePath,fileName = os.path.split(file)
		if 1 == imgNChannels:
			testImg = cv.imread(file,0)
		else:
			testImg = cv.imread(file)

		ResizeImg_temp = cv.resize(testImg,(imgCols,imgRows))
		ResizeImg = ResizeImg_temp[:,int(-ResizeImg_temp.shape[1]/1.75-1):-1]

		testData_x =  np.asarray(ResizeImg,dtype='float32').flatten(0)/256.0
		testData_x = (testData_x - x_mean).reshape((1,testData_x.shape[0]))

		# carRecog = theano.function(
		# 	[x],
		# 	layer3_input,
		# )


		classID = carRecog(testData_x)

		i = i+1
		print i

		# diff = numpy.dot((classID - objectID),(classID-objectID).T)
		diff = 0
		for k in xrange(classID.shape[1]):
			value = classID[0,k] - objectID[0,k]
			if value*value > 0.001:
				diff += value*value


		if  ( diff < threshold ):
			p = p+1

			# 删除原图
			deleteSrc = 0

			endP = string.rfind(fileName,'_')
			firstName = fileName[:endP]

			imResult = cv.imread(searchPath +'/'+firstName+'_result.jpg')
			imSrc = cv.imread(searchPath + '/'+firstName+'_src.jpg')
			imHead = cv.imread(searchPath + '/'+firstName+'_head.bmp')

			# 保存图片
			saveFirstName = '%d' % int(( diff )*100000)
			cv.imwrite(destPath + '/'+ saveFirstName+'_'+firstName+'_result.jpg',imResult)
			cv.imwrite(destPath + '/'+saveFirstName+'_'+firstName+'_src.jpg',imSrc)
			cv.imwrite(destPath + '/'+saveFirstName+'_'+firstName+'_head.bmp',imHead)


			# cv.imshow("head",testImg)
			# cv.imshow("result",imResult)
			# cv.imshow("src",imSrc)
			# cv.waitKey(0)
			# cv.imshow("result",testImg)




		# cv.waitKey(0)

	print p
	print i
	# 	else:
	# 		errorNum += 1
	# 		cv.imshow("inputImg",ResizeImg)
	# 		cv.waitKey(10)
	#
	# print ('correctNum: %i \nerrorNum: %i\ntotal: %i' % (correctNum,errorNum,correctNum+errorNum))
	# print ('correctRate: %f %%' % (100*correctNum*1.0/(correctNum+errorNum)))
	# print ('smallCarRate: %f' % (smallCar*1.0/(bigCar+smallCar)))


if __name__ == '__main__':
	theano.config.floatX = 'float32'
	print theano.config.floatX
	evaluate_carModel()