#coding:utf-8#
__author__ = 'ASUS'
import numpy as np
import  cv2 as cv
import cPickle
import os

if __name__ == '__main__':

	in_dir = 'H:\\veheicleSample\\headOfCar\\samplePKL\\detailClass\\'
	dir_out = in_dir
	dataName = 'HalfGrayFlip_random50.pkl'
	inDataPath = in_dir + dataName

	PCADataName = 'PCAData2.pkl'

	srcData_f = open(inDataPath,'rb')
	train_set,valid_set,test_set = cPickle.load(srcData_f)

	trainData_x,trainData_y = train_set
	validData_x,validData_y = valid_set
	testData_x,testData_y = test_set

	trainData_x =  np.asarray(trainData_x,dtype='float32')/256.0
	validData_x =  np.asarray(validData_x,dtype='float32')/256.0
	testData_x =  np.asarray(testData_x,dtype='float32')/256.0

	if (  1 or not os.path.isfile(dir_out+'/'+PCADataName)):


		# V,S,mean_x = pca.pca(trainData_x)
		# epsilon = 0.1
		# U = V.T
		# ZCAWhite = U.dot( np.diag((1.0/np.sqrt(S+epsilon))).dot(U.T))  #U*(1/sqrt(s+epsilon))*U'

		trainData_x = validData_x
		trainData_xmean = (trainData_x.mean(axis = 0)).reshape(1,(trainData_x.shape[1]))
		meanData , eigenvector = cv.PCACompute(trainData_x,trainData_xmean)

		saveFile = open(dir_out + '\\' + PCADataName,'wb')
		cPickle.dump( [meanData,eigenvector],saveFile)
		saveFile.close()

	else:
		saveFile = open(dir_out + '\\' + PCADataName,'rb')
		meanData , eigenvector = cPickle.load(saveFile)
		print 1




	trainData_x = cv.PCAProject(trainData_x,meanData,eigenvector)
	validData_x = cv.PCAProject(validData_x,meanData,eigenvector)
	testData_x = cv.PCAProject(testData_x,meanData,eigenvector)



		# num_data,dim = trainData_x.shape
		#
		# # dim = 800
		# x = trainData_x[0:dim]
		# x = x.T
		# mean_x = x.mean(axis = 0)
		# x0 = x - mean_x  #tile 整块扩展矩阵
		# sigma = np.dot(x0,x0.T)/num_data
		# U,S,V = np.linalg.svd(sigma)
		# epsilon = 0.1
		# ZCAWhite = U.dot( np.diag((1.0/np.sqrt(S+epsilon))).dot(U.T))  #U*(1/sqrt(s+epsilon))*U'
		#
		# # 保存白化矩阵，在测试时需要使用相同的白化矩阵对测试图片进行白化
		# saveFile = open(dir_out +'/'+eigenvectorName,'wb')
		# cPickle.dump(ZCAWhite,saveFile)
		# saveFile.close()
		# print 'ZCAdone!'
		#
		# x_mean = trainData_x.mean(0)
		# saveFile = open(dir_out +'/'+ meanDataName,'wb')
		# cPickle.dump(x_mean,saveFile)
		# saveFile.close()

	# else:
	# 	meanData_f = open(dir_out +'/'+ meanDataName,'rb')
	# 	ZCAData_f = open(dir_out +'/'+ ZCADataName,'rb')
	#
	# 	x_mean = cPickle.load(meanData_f)
	# 	ZCAWhite = cPickle.load(ZCAData_f)
	# 	meanData_f.close()
	#
	# trainData_x = np.dot((trainData_x - trainData_x.mean()) ,ZCAWhite)
	# trainData_x = trainData_x - x_mean
	# trainData_y =  trainData_y.flatten()
	#
	# validData_x = np.dot((validData_x - trainData_x.mean()) ,ZCAWhite)
	# validData_x = validData_x - x_mean
	# validData_y =  validData_y.flatten()
	#
	# testData_x = np.dot((testData_x - trainData_x.mean()) ,ZCAWhite)
	# testData_x = testData_x - x_mean
	# testData_y =  testData_y.flatten()
	#
	# train_set_x, train_set_y = shared_dataset(trainData_x,trainData_y)
	# valid_set_x, valid_set_y = shared_dataset(validData_x,validData_y)
	# test_set_x, test_set_y = shared_dataset(testData_x,testData_y)
	#
	# rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),	(test_set_x, test_set_y)]
	# return rval