#coding: gb18030
#coding:utf8#
# -*-coding:utf8-*-#

__author__ = 'ASUS'

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



def load_data(dataset_path):

	read_file = open(dataset_path,'rb')
	datas,labels = cPickle.load(read_file)
	read_file.close()

	# def shared_dataset(data_x, data_y, borrow=True):
	# 	shared_x = theano.shared(numpy.asarray(data_x,
	# 										   dtype=theano.config.floatX),
	# 							 borrow=borrow)
	# 	shared_y = theano.shared(numpy.asarray(data_y,
	# 										   dtype=theano.config.floatX),
	# 							 borrow=borrow)
	# 	return shared_x, T.cast(shared_y, 'int32')


	# x, y = shared_dataset(datas,labels)
	x, y = (datas,labels)
	return x, y

def load_params(params_file):
    f=open(params_file,'rb')
    layer0_params=cPickle.load(f)
    layer1_params=cPickle.load(f)
    layer2_params=cPickle.load(f)
    layer3_params=cPickle.load(f)
    f.close()
    return layer0_params,layer1_params,layer2_params,layer3_params


def shared_dataset(data_x, data_y, borrow=True):
	shared_x = theano.shared(np.asarray(data_x,
										   dtype=theano.config.floatX),
							 borrow=borrow)
	shared_y = theano.shared(np.asarray(data_y,
										   dtype=theano.config.floatX),
							 borrow=borrow)
	return shared_x, T.cast(shared_y, 'int32')




def readAndWhiteData(inDataPath,dir_out):

	#将所有种类的车辆样本集中到一个文件数据，并打乱其顺序

	# vehicleType = '3_benz'
	meanDataName = 'x_mean.pkl'
	ZCADataName = 'ZCAData.pkl'

	srcData_f = open(inDataPath,'rb')
	train_set,valid_set,test_set = cPickle.load(srcData_f)

	trainData_x,trainData_y = train_set
	validData_x,validData_y = valid_set
	testData_x,testData_y = test_set

	trainData_x =  np.asarray(trainData_x,dtype='float32')/256.0
	validData_x =  np.asarray(validData_x,dtype='float32')/256.0
	testData_x =  np.asarray(testData_x,dtype='float32')/256.0

	if ( not os.path.isfile(dir_out+'/'+meanDataName)):

	#对样本进行ZCA白化

		# V,S,mean_x = pca.pca(trainData_x)
		# epsilon = 0.1
		# U = V.T
		# ZCAWhite = U.dot( np.diag((1.0/np.sqrt(S+epsilon))).dot(U.T))  #U*(1/sqrt(s+epsilon))*U'


		# num_data,dim = trainData_x.shape
		# # dim = 800
		# x = trainData_x[0:dim]
		# x = x.T
		# mean_x = x.mean(axis = 0)
		# x0 = x - mean_x  #tile 整块扩展矩阵
		# sigma = numpy.dot(x0,x0.T)/num_data
		# U,S,V = numpy.linalg.svd(sigma)
		# epsilon = 0.1
		# ZCAWhite = U.dot( np.diag((1.0/np.sqrt(S+epsilon))).dot(U.T))  #U*(1/sqrt(s+epsilon))*U'
		#
		# # 保存白化矩阵，在测试时需要使用相同的白化矩阵对测试图片进行白化
		# saveFile = open(dir_out +'/'+ZCADataName,'wb')
		# cPickle.dump(ZCAWhite,saveFile)
		# saveFile.close()
		# print 'ZCAdone!'

		x_mean = trainData_x.mean(0)
		saveFile = open(dir_out +'/'+ meanDataName,'wb')
		cPickle.dump(x_mean,saveFile)
		saveFile.close()

	else:
		meanData_f = open(dir_out +'/'+ meanDataName,'rb')
		x_mean = cPickle.load(meanData_f)
		meanData_f.close()

		# ZCAData_f = open(dir_out +'/'+ ZCADataName,'rb')
		# ZCAWhite = cPickle.load(ZCAData_f)
		# ZCAData_f.close()
		# trainData_x = np.dot((trainData_x - trainData_x.mean()) ,ZCAWhite)
		# validData_x = np.dot((validData_x - trainData_x.mean()) ,ZCAWhite)
		# testData_x = np.dot((testData_x - trainData_x.mean()) ,ZCAWhite)

	# trainData_x = trainData_x - x_mean
	trainData_y =  trainData_y.flatten()

	# validData_x = validData_x - x_mean
	validData_y =  validData_y.flatten()

	# testData_x = testData_x - x_mean
	testData_y =  testData_y.flatten()

	train_set_x, train_set_y = shared_dataset(trainData_x,trainData_y)
	valid_set_x, valid_set_y = shared_dataset(validData_x,validData_y)
	test_set_x, test_set_y = shared_dataset(testData_x,testData_y)

	rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),	(test_set_x, test_set_y)]
	return rval

def ReLU(x):
	return (x + abs(x))/2.0

#分类器，即CNN最后一层，采用逻辑回归（softmax）
class LogisticRegression(object):
	def __init__(self, input,params_W,params_b,usePreParams, n_in, n_out):
		if usePreParams:
			self.W = params_W
			self.b = params_b

		else:
			self.W = theano.shared(
				value=numpy.zeros(
					(n_in, n_out),
					dtype=theano.config.floatX
					),
				name='W',
				borrow=True
				)
			self.b = theano.shared(
					value=numpy.zeros(
					(n_out,),
					dtype=theano.config.floatX
					),
				name='b',
				borrow=True
		)
		# s = input.get_value(borrow=True).shape
		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
		self.y_pred = T.argmax(self.p_y_given_x, axis=1)
		self.params = [self.W, self.b]

	def negative_log_likelihood(self, y):
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

	def errors(self, y):
		if y.ndim != self.y_pred.ndim:
			raise TypeError(
				'y should have the same shape as self.y_pred',
				('y', y.type, 'y_pred', self.y_pred.type)
			)
		if y.dtype.startswith('int'):
			# return (T.mean(T.neq(self.y_pred, y)),T.sum(T.neq(self.y_pred, y)),T.sum(T.eq(self.y_pred, y)))
			return T.mean(T.neq(self.y_pred, y))
		else:
			raise NotImplementedError()


class HiddenLayer(object):
	def __init__(self, rng, input,params_W,params_b,usePreParams, n_in, n_out, W=None, b=None,
				 activation=ReLU):

		self.input = input
		if usePreParams:
			self.W = params_W
			self.b = params_b
		else:
			if W is None:
				W_values = numpy.asarray(
					rng.uniform(
						low=-numpy.sqrt(6. / (n_in + n_out)),
						high=numpy.sqrt(6. / (n_in + n_out)),
						size=(n_in, n_out)
					),
					dtype=theano.config.floatX
				)
				if activation == theano.tensor.nnet.sigmoid:
					W_values *= 4
				W = theano.shared(value=W_values, name='W', borrow=True)

			if b is None:
				b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
				b = theano.shared(value=b_values, name='b', borrow=True)

			self.W = W
			self.b = b

		lin_output = T.dot(input, self.W) + self.b
		self.output = (
			lin_output if activation is None
			else activation(lin_output)
		)
		# parameters of the model
		self.params = [self.W, self.b]



class LeNetConvPoolLayer(object):

	def __init__(self, rng, input ,params_W,params_b,usePreParams,filter_shape, image_shape, poolsize=(2, 2)):

		assert image_shape[1] == filter_shape[1]
		self.input = input

		fan_in = numpy.prod(filter_shape[1:])
		fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
				   numpy.prod(poolsize))

		if usePreParams:
			self.W = params_W
			self.b = params_b

		else:
			# initialize weights with random weights
			W_bound = numpy.sqrt(6. / (fan_in + fan_out))
			self.W = theano.shared(
				numpy.asarray(
					rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
					dtype=theano.config.floatX
				),
				borrow=True
			)

			# the bias is a 1D tensor -- one bias per output feature map
			b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
			self.b = theano.shared(value=b_values, borrow=True)

		# 卷积
		conv_out = conv.conv2d(
			input=input,
			filters=self.W,
			filter_shape=filter_shape,
			image_shape=image_shape
		)

		# 子采样
		pooled_out = downsample.max_pool_2d(
			input=conv_out,
			ds=poolsize,
			ignore_border=True
		)

		# self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
		# self.output = T.nnet.sigmoid(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
		self.output = ReLU(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))


		# store parameters of this layer
		self.params = [self.W, self.b]


def save_params(fileName,param1,param2,param3,param4):
        import cPickle
        write_file = open(fileName, 'wb')
        cPickle.dump(param1, write_file, -1)
        cPickle.dump(param2, write_file, -1)
        cPickle.dump(param3, write_file, -1)
        cPickle.dump(param4, write_file, -1)
        write_file.close()


def trainCarClassifier(learning_rate=0.04, n_epochs=1000, nkerns=[10,20],
					  batch_size = 100, usePreParams=True):

	in_dir = 'H:\\veheicleSample\\headOfCar\\samplePKL\\detailClass\\'
	dataName = 'HalfGrayFlipEqhist_random50.pkl'
	inDataPath = in_dir + dataName

	out_dir = 'H:\\veheicleSample\\headOfCar\\samplePKL\\detailClass'
	imgNChannels = 1
	imgRows = 98
	imgCols = 98


	filterSize = [11,7]  # 卷积核宽度
	L0PoolSize = (4,4)
	L1PoolSize = (2,2)
	HL_nout = 500  #隐层输入
	classNum = 0  #分类数量

	vehicleType = '50class'
	# scale = 9.0/14.0     #训练样本与测试样本的比例
	saveParamsName = vehicleType+'_params.pkl'
	# add_negat = 0;   #改   ###########

	if usePreParams:
		best_validation_loss = 18/100.0  #test 2.848485
	else:
		best_validation_loss = numpy.inf

	dataSets = readAndWhiteData(inDataPath,out_dir)
	train_set_x,train_set_y = dataSets[0]
	valid_set_x,valid_set_y = dataSets[1]  ###########
	test_set_x,test_set_y = dataSets[2]
	# classNum = (T.max(test_set_y)).get_value(borrow=True) + 1
	classNum = np.max(test_set_y.eval(),0) + 1

	n_train_batches = train_set_x.get_value(borrow=True).shape[0]
	n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
	n_test_batches = test_set_x.get_value(borrow=True).shape[0]

	# validBatchSize = batch_size#valid_set_x.get_value(borrow=True).shape[0]
	n_train_batches /= batch_size
	n_valid_batches /= batch_size
	n_test_batches /= batch_size

	print '... building the model'

	index = T.lscalar()
	x = T.matrix('x')
	y = T.ivector('y')

	# valid_x = T.matrix('valid_x')
	# valid_y = T.ivector('valid_y')

	if usePreParams:
		layer0_params,layer1_params,layer2_params,layer3_params=load_params(saveParamsName)
	else:
		layer0_params,layer1_params,layer2_params,layer3_params=[[0,0],[0,0],[0,0],[0,0]]


	rng = numpy.random.RandomState(23455)
	layer0_input = x.reshape((batch_size,imgNChannels,imgRows,imgCols))
	layer0 = LeNetConvPoolLayer(
		rng,
		input = layer0_input,
        params_W=layer0_params[0],
        params_b=layer0_params[1],
		usePreParams=usePreParams,
		image_shape = (batch_size,imgNChannels,imgRows,imgCols),
		filter_shape = (nkerns[0],imgNChannels,filterSize[0],filterSize[0]),
		poolsize=L0PoolSize
	)


	# 第二个卷积+maxpool层,输入是上层的输出，即(batch_size, nkerns[0], 26, 21)
	L1_imgShapeRows = (imgRows - filterSize[0]+1)/L0PoolSize[0]
	L1_imgShapeCols = (imgCols - filterSize[0]+1)/L0PoolSize[0]
	layer1 = LeNetConvPoolLayer(
		rng,
		input=layer0.output,
		params_W=layer1_params[0],
        params_b=layer1_params[1],
		usePreParams=usePreParams,
		image_shape=(batch_size, nkerns[0],L1_imgShapeRows,L1_imgShapeCols),
		filter_shape=(nkerns[1], nkerns[0], filterSize[1], filterSize[1]),
		poolsize=L1PoolSize
	)

	layer2_input = layer1.output.flatten(2)
	HL_nout = HL_nout
	L2_imgShapeRows = (L1_imgShapeRows - filterSize[1] + 1)/L1PoolSize[0]
	L2_imgShapeCols = (L1_imgShapeCols - filterSize[1] + 1)/L1PoolSize[0]
	layer2 = HiddenLayer(
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
	layer3 = LogisticRegression(
		input=layer3_input,
		params_W=layer3_params[0],
        params_b=layer3_params[1],
		usePreParams=usePreParams,
		n_in=HL_nout, n_out=classNum)


	test_model = theano.function(
		[index],
		layer3.errors(y),
		givens={
			x: test_set_x[index * batch_size: (index + 1) * batch_size],
			y: test_set_y[index * batch_size: (index + 1) * batch_size]
		}
	)

	validate_model = theano.function(
		[index],
		layer3.errors(y),
		givens={
			x: valid_set_x[index * batch_size: (index + 1) * batch_size],
			y: valid_set_y[index * batch_size: (index + 1) * batch_size]
		}
	)

	trainerror_model = theano.function(
		[index],
		layer3.errors(y),
		givens={
			x: train_set_x[index * batch_size: (index + 1) * batch_size],
			y: train_set_y[index * batch_size: (index + 1) * batch_size]
		}
	)

	# 所有参数
	params = layer3.params + layer2.params + layer1.params + layer0.params

	L1  = T.sum(abs(layer3.params[0]))+T.sum(abs(layer2.params[0]))+T.sum(abs(layer1.params[0]))+T.sum(abs(layer0.params[0]))
	L2_sqr = T.sum(layer3.params[0] ** 2)+T.sum(layer2.params[0] ** 2)+T.sum(layer1.params[0] ** 2)+T.sum(layer0.params[0] ** 2)

	#L2应除以 w变量个数2倍
	# L1 = abs(layer3.W).sum()+abs(layer2.W).sum()+abs(layer1.W).sum()+abs(layer0.W).sum()
	# L2_sqr = (layer3.W**2).sum() + (layer2.W**2).sum()+ (layer1.W**2).sum() + (layer0.W**2).sum()

	lamda1 =  0.001  #small than 0.001
	lamda2 =  0
	cost = layer3.negative_log_likelihood(y)+lamda1*L1+lamda2*L2_sqr
	# cost = layer3.negative_log_likelihood(y)
	#各个参数的梯度
	grads = T.grad(cost, params)
	#参数更新规则
	updates = [
		(param_i, param_i - learning_rate * grad_i)
		for param_i, grad_i in zip(params, grads)
	]
	#train_model在训练过程中根据MSGD优化更新参数
	train_model = theano.function(
		[index],
		cost,
		updates=updates,
		givens={
			x: train_set_x[index * batch_size: (index + 1) * batch_size],
			y: train_set_y[index * batch_size: (index + 1) * batch_size]
			},
		# mode = 'DebugMode'
	)


	###############
	# 训练CNN阶段，寻找最优的参数。
	###############
	print '... training'
	#在LeNet5中，batch_size=500,n_train_batches=50000/500=100，patience=10000
	#在olivettifaces中，batch_size=40,n_train_batches=320/40=8, paticence可以相应地设置为800，这个可以根据实际情况调节，调大一点也无所谓
	patience = 800000
	patience_increase = 2
	improvement_threshold = 0.99
	validation_frequency = min(n_train_batches, patience / 2)

	best_iter = 0
	test_score = 0.
	start_time = time.clock()

	epoch = 0
	done_looping = False

	while (epoch < n_epochs) and (not done_looping):
		epoch = epoch + 1
		for minibatch_index in xrange(n_train_batches):
			iter = (epoch - 1) * n_train_batches + minibatch_index  #iter表示执行batch的次数
			if iter % 100 == 0:
				print 'training @ iter = ', iter
			cost_ij = train_model(minibatch_index)  #???????

			if (iter + 1) % validation_frequency == 0:

				# compute zero-one loss on validation set
				validation_losses = [validate_model(i) for i
									 in xrange(n_valid_batches)]

				if 0 == epoch%10:
					trainerror_losses = [trainerror_model(i) for i
									 in xrange(n_train_batches)]
				# validation_losses,errorNum,correctNum = validate_model(0)
				else:
					trainerror_losses = 100

				this_validation_loss = numpy.mean(validation_losses)
				this_train_loss = numpy.mean(trainerror_losses)

				print('epoch %i, minibatch %i/%i, trainError=%.4f validError=%.4f %%' %
					  (epoch, minibatch_index + 1, n_train_batches,
					   this_train_loss * 100.,
					   this_validation_loss * 100.
					   ,
					  ))
				print cost_ij

				# if we got the best validation score until now
				if this_validation_loss < best_validation_loss:

					#improve patience if loss improvement is good enough
					if this_validation_loss < best_validation_loss * improvement_threshold:
						patience = max(patience, iter * patience_increase)

					# save best validation score and iteration number
					best_validation_loss = this_validation_loss
					best_iter = iter
					save_params(saveParamsName,layer0.params,layer1.params,layer2.params,layer3.params) #保存参数
					#  test it on the test set
					test_losses = [
						test_model(i)
						for i in xrange(n_test_batches)
					]
					test_score = numpy.mean(test_losses)
					print(('	  ------epoch %i, minibatch %i/%i, test error of '
						   'best model %f %%') %
						  (epoch, minibatch_index + 1, n_train_batches,
						   test_score * 100.))

				if 0 == epoch%10:
					save_params('temp'+saveParamsName,layer0.params,layer1.params,layer2.params,layer3.params) #保存参数
			if patience <= iter:
				done_looping = True
				break

	end_time = time.clock()
	print('Optimization complete.')
	print('Best validation score of %f %% obtained at iteration %i, '
		  'with test performance %f %%' %
		  (best_validation_loss * 100., best_iter + 1, test_score * 100.))
	print >> sys.stderr, ('The code for file ' +
						  os.path.split(__file__)[1] +
						  ' ran for %.2fm' % ((end_time - start_time) / 60.))



if __name__ == '__main__':
	trainCarClassifier()