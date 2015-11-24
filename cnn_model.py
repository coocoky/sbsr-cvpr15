# import cPickle
# import gzip
# import os
# import sys
# import time
# import scipy.io as sio

import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.tensor.sort import sort, argsort

from cnn_setting import CNNSetting


class ConvPoolLayer(object):

    def __init__(self, rng, filter_shape, image_shape,
                 poolsize=(2, 2), activation=lambda x: T.maximum(0.0, x), \
                 use_dropout=False, dropout_p=0.5):

        assert image_shape[1] == filter_shape[1]

        self.image_shape = image_shape
        self.filter_shape = filter_shape
        self.poolsize = poolsize
        self.activation = activation
        self.use_dropout = use_dropout
        self.dropout_p = dropout_p

        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out)) * 0.5
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        self.params = [self.W, self.b]


    def process(self, image):

        conv_out = conv.conv2d(input=image, filters=self.W,
                filter_shape=self.filter_shape, image_shape=self.image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=self.poolsize, ignore_border=True)

        self.lin_output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.output = self.activation(self.lin_output)

        if self.use_dropout:
            srng = theano.tensor.shared_randomstreams.RandomStreams(
                    self.rng.randint(999999))
            # p=1-p because 1's indicate keep and p is prob of dropping
            mask = srng.binomial(n=1, p=1-self.dropout_p, size=self.output.shape)
            self.output = self.output * T.cast(mask, theano.config.floatX)

        return self.output


class LRLayer(object):

    def __init__(self, rng, n_in, n_out, regression_flag=False, active_func=None, \
                 use_dropout=False, dropout_p=0.5):

        self.rng = rng
        # W_value = self.rng.standard_normal(size=(n_in, n_out))
        W_value = self.rng.standard_normal(size=(n_in, n_out)) * numpy.sqrt( 6.0 / (n_in+n_out) )
        self.W = theano.shared(value=numpy.asarray(W_value, dtype=theano.config.floatX),
                                name='W', borrow=True)
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)

        self.regression_flag = regression_flag
        self.active_func = active_func
        self.use_dropout = use_dropout
        self.dropout_p = dropout_p
        self.p_y_given_x = None
        self.y_pred = None

        # parameters of the model
        self.params = [self.W, self.b]


    def process(self, input):
        self.lin_output = T.dot(input, self.W) + self.b
        if self.active_func == 'relu':
            self.output = T.maximum(0.0, self.lin_output)
        elif self.active_func == 'tanh':
            self.output = T.tanh(self.lin_output)
        else:
            # self.active_func is None:
            self.output = self.lin_output

        if self.regression_flag:
            self.p_y_given_x = T.nnet.softmax(self.lin_output)
            self.y_pred = T.argmax(self.p_y_given_x, axis=1)
            return self.p_y_given_x

        elif self.use_dropout:
            srng = theano.tensor.shared_randomstreams.RandomStreams(
                    self.rng.randint(999999))
            # p=1-p because 1's indicate keep and p is prob of dropping
            mask = srng.binomial(n=1, p=1-self.dropout_p, size=self.output.shape)
            self.output = self.output * T.cast(mask, theano.config.floatX)

        return self.output


class CNN_model(object):

    def __init__(self, prm, input_shape, setting):
        self.rng = prm.prng
        self.setting = setting

        batch_size = setting.batch_size
        nchannel = setting.nchannel
        L        = setting.n_layer
        nkerns   = setting.nkerns[:]
        nkerns.append(setting.nout)
        fs       = setting.fs
        ps       = setting.ps
        ptsize = (input_shape[0], input_shape[1])
        otsize = setting.output_map_size(ptsize)

        self.layer = [None] * L
        self.params = []
        self.has_hidden_layer = False

        n_maps = nchannel
        for i in xrange(L):
            if setting.tag[i] == 'c': # convolutonal layer (with pooling)
                filter_shape = (nkerns[i], n_maps, fs[i][0], fs[i][1])
                image_shape  = (batch_size, n_maps, otsize[i][0], otsize[i][1])

                self.layer[i] = ConvPoolLayer(self.rng, filter_shape, image_shape, ps[i])

            elif setting.tag[i] == 'h': # fully connected hidden layer
                self.has_hidden_layer = True
                self.layer[i] = LRLayer(self.rng, n_in=n_maps*otsize[i][0]*otsize[i][1], \
                                        n_out=nkerns[i], regression_flag=False, active_func='relu', \
                                        use_dropout=setting.use_dropout, dropout_p=setting.dropout_p)

            elif setting.tag[i] == 'f': # fully connected final layer
                self.layer[i] = LRLayer(self.rng, n_in=n_maps*otsize[i][0]*otsize[i][1], \
                                        n_out=nkerns[i], regression_flag=False, active_func=setting.final_active_func)

            self.params += self.layer[i].params
            n_maps = nkerns[i]


    def process(self, input):
        setting = self.setting
        L = setting.n_layer

        output = [None] * L
        featmap = input

        for i in xrange(L):
            if setting.tag[i] == 'h' or setting.tag[i] == 'f':
                featmap = featmap.flatten(2)

            output[i] = self.layer[i].process(featmap)
            featmap = output[i]

        return featmap, output


class pairwise_model():

    def __init__(self, prm, ishape, cnn_setting):

        [cnn_setting_ss, cnn_setting_vs] = cnn_setting
        sketch_model = CNN_model(prm, ishape, cnn_setting_ss)
        view_model   = CNN_model(prm, ishape, cnn_setting_vs)

        self.setting = cnn_setting_ss
        self.sketch_model = sketch_model
        self.view_model = view_model
        self.params = sketch_model.params + view_model.params


def gen_models(prm, cnn_setting, inputWH, params_init_val=None):

    print 'generating model ... \n'

    ishape = (inputWH, inputWH)  # this is the size of MNIST images
    cnn_model = CNN_model(prm, ishape, cnn_setting)
    input_shp = (cnn_setting.batch_size, cnn_setting.nchannel, inputWH, inputWH)

    params = cnn_model.params
    if params_init_val: #got initialization values
        for k in xrange(len(params)):
            params[k].set_value(params_init_val[k])

    batch_size = cnn_setting.batch_size
    learning_rate = prm.learning_rate
    lr_fc = prm.lr_fc
    lr_ff = prm.lr_ff

    # allocate symbolic variables for the data
    x0 = T.matrix('x0')   # the data is presented as rasterized images
    x1 = T.matrix('x1')
    p0 = T.matrix('p0')
    p1 = T.matrix('p1')
    ys = T.ivector('ys')  # class label of sketch
    yv = T.ivector('yv')  # class label of view

    sketch_code, sketch_output = cnn_model.process(x0.reshape(input_shp))
    sketch_peer_code, sketch_peer_output = cnn_model.process(p0.reshape(input_shp))
    view_code, view_output  = cnn_model.process(x1.reshape(input_shp))
    view_peer_code, view_peer_output  = cnn_model.process(p1.reshape(input_shp))

    Cpos = cnn_setting.Cpos
    Cneg = cnn_setting.Cneg
    alpha = cnn_setting.alpha
    sw = cnn_setting.sw
    vw = cnn_setting.vw

    # distance-matrix
    y_mat = T.neq(ys.dimshuffle(0,'x'), yv.dimshuffle('x',0))

    srng = theano.tensor.shared_randomstreams.RandomStreams(
            prm.prng.randint(8888))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask_pos = T.cast(srng.binomial(n=1, p=prm.ppos, size=y_mat.shape), theano.config.floatX)
    mask_neg = T.cast(srng.binomial(n=1, p=prm.pneg, size=y_mat.shape), theano.config.floatX)


    numpos = T.sum(T.eq(y_mat, 0)*mask_pos)
    numneg = T.sum(T.eq(y_mat, 1)*mask_neg)

    l1norm = T.sum(T.abs_(sketch_code.dimshuffle(0, 'x', 1) - view_code.dimshuffle('x', 0, 1)), axis=2)
    diff = l1norm ** 2
    dpos = Cpos * diff * T.eq(y_mat, 0)
    dneg = Cneg * T.exp(-alpha / Cneg * l1norm) * T.eq(y_mat, 1)

    l1norm_sp = T.sum(T.abs_(sketch_code.dimshuffle(0, 'x', 1) - sketch_peer_code.dimshuffle('x', 0, 1)), axis=2)
    diff_sp = l1norm_sp ** 2
    dpos_sp = sw * Cpos * diff_sp * T.eq(y_mat, 0)
    dneg_sp = Cneg * T.exp(-alpha / Cneg * l1norm_sp) * T.eq(y_mat, 1)

    l1norm_vp = T.sum(T.abs_(view_code.dimshuffle(0, 'x', 1) - view_peer_code.dimshuffle('x', 0, 1)), axis=2)
    diff_vp = l1norm_vp ** 2
    dpos_vp = vw * Cpos * diff_vp * T.eq(y_mat, 0)
    dneg_vp = Cneg * T.exp(-alpha / Cneg * l1norm_vp) * T.eq(y_mat, 1)

    cost_pos = T.sum((dpos + dpos_sp + dpos_vp) * mask_pos) / prm.fpos
    cost_neg = T.sum((dneg + dneg_sp + dneg_vp) * mask_neg) / prm.fneg
    cost = cost_pos + cost_neg

    posdiff = T.sum(l1norm * T.eq(y_mat, 0) * mask_pos) / numpos
    negdiff = T.sum(l1norm * T.eq(y_mat, 1) * mask_neg) / numneg

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    L = cnn_setting.n_layer
    updates = []

    for i in xrange(L):
        if cnn_setting.tag[i] == 'c':
            updates.append((params[i*2], params[i*2] - lr_fc*learning_rate * grads[i*2]))
            updates.append((params[i*2+1], params[i*2+1] - lr_fc*learning_rate * grads[i*2+1]))
        else:
            updates.append((params[i*2], params[i*2] - lr_ff*learning_rate * grads[i*2]))
            updates.append((params[i*2+1], params[i*2+1] - lr_ff*learning_rate * grads[i*2+1]))


    cnn_model.x0 = x0
    cnn_model.x1 = x1
    cnn_model.p0 = p0
    cnn_model.p1 = p1
    # cnn_model.y  = y
    cnn_model.ys = ys
    cnn_model.yv = yv
    cnn_model.cost = cost
    cnn_model.cost_pos = cost_pos
    cnn_model.cost_neg = cost_neg

    cnn_model.dpos = T.sum(dpos * mask_pos)
    cnn_model.dneg = T.sum(dneg * mask_neg)
    cnn_model.dpos_sp = T.sum(dpos_sp * mask_pos)
    cnn_model.dneg_sp = T.sum(dneg_sp * mask_neg)
    cnn_model.dpos_vp = T.sum(dpos_vp * mask_pos)
    cnn_model.dneg_vp = T.sum(dneg_vp * mask_neg)

    cnn_model.numpos = numpos
    cnn_model.numneg = numneg
    cnn_model.l1 = l1norm
    cnn_model.posdiff = posdiff
    cnn_model.negdiff = negdiff

    cnn_model.grads = grads
    cnn_model.updates = updates

    cnn_model.update_params = params

    cnn_model.sketch_code = sketch_code
    cnn_model.sketch_output = sketch_output
    cnn_model.view_code = view_code
    cnn_model.view_output = view_output

    cnn_model.sketch_peer_output = sketch_peer_output
    cnn_model.view_peer_output = view_peer_output

    return cnn_model



def gen_dual_models(prm, cnn_setting, inputWH, params_init_val=None):

    print 'generating model ... \n'

    ishape = (inputWH, inputWH)  # this is the size of MNIST images
    cnn_model = pairwise_model(prm, ishape, cnn_setting)

    cnn_setting_ss, cnn_setting_vs = cnn_setting

    input_shp_ss = (cnn_setting_ss.batch_size, cnn_setting_ss.nchannel, inputWH, inputWH)
    input_shp_vs = (cnn_setting_vs.batch_size, cnn_setting_vs.nchannel, inputWH, inputWH)

    if params_init_val: #got initialization values
        for k in xrange(len(cnn_model.params)):
            cnn_model.params[k].set_value(params_init_val[k])

    params = cnn_model.params

    batch_size = cnn_setting_ss.batch_size
    learning_rate = cnn_setting_ss.learning_rate

    # allocate symbolic variables for the data
    x0 = T.matrix('x0')   # the data is presented as rasterized images
    x1 = T.matrix('x1')
    p0 = T.matrix('p0')
    p1 = T.matrix('p1')
    ys = T.ivector('ys')  # class label of sketch
    yv = T.ivector('yv')  # class label of view

    sketch_code, sketch_output = cnn_model.sketch_model.process(x0.reshape(input_shp_ss))
    sketch_peer_code, sketch_peer_output = cnn_model.sketch_model.process(p0.reshape(input_shp_ss))
    view_code, view_output  = cnn_model.view_model.process(x1.reshape(input_shp_vs))
    view_peer_code, view_peer_output  = cnn_model.view_model.process(p1.reshape(input_shp_vs))

    Cpos = cnn_setting_ss.Cpos
    Cneg = cnn_setting_ss.Cneg
    alpha = cnn_setting_ss.alpha
    sw = cnn_setting_ss.sw
    vw = cnn_setting_ss.vw

    # distance-matrix
    y_mat = T.neq(ys.dimshuffle(0,'x'), yv.dimshuffle('x',0))

    srng = theano.tensor.shared_randomstreams.RandomStreams(
            prm.prng.randint(8888))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask_pos = T.cast(srng.binomial(n=1, p=prm.ppos, size=y_mat.shape), theano.config.floatX)
    mask_neg = T.cast(srng.binomial(n=1, p=prm.pneg, size=y_mat.shape), theano.config.floatX)

    numpos = T.sum(T.eq(y_mat, 0)*mask_pos)
    numneg = T.sum(T.eq(y_mat, 1)*mask_neg)

    l1norm = T.sum(T.abs_(sketch_code.dimshuffle(0, 'x', 1) - view_code.dimshuffle('x', 0, 1)), axis=2)
    diff = l1norm ** 2
    dpos = Cpos * diff * T.eq(y_mat, 0)
    dneg = Cneg * T.exp(-alpha / Cneg * l1norm) * T.eq(y_mat, 1)

    l1norm_sp = T.sum(T.abs_(sketch_code.dimshuffle(0, 'x', 1) - sketch_peer_code.dimshuffle('x', 0, 1)), axis=2)
    diff_sp = l1norm_sp ** 2
    dpos_sp = sw * Cpos * diff_sp * T.eq(y_mat, 0)
    dneg_sp = Cneg * T.exp(-alpha / Cneg * l1norm_sp) * T.eq(y_mat, 1)

    l1norm_vp = T.sum(T.abs_(view_code.dimshuffle(0, 'x', 1) - view_peer_code.dimshuffle('x', 0, 1)), axis=2)
    diff_vp = l1norm_vp ** 2
    dpos_vp = vw * Cpos * diff_vp * T.eq(y_mat, 0)
    dneg_vp = Cneg * T.exp(-alpha / Cneg * l1norm_vp) * T.eq(y_mat, 1)

    cost_pos = T.sum((dpos + dpos_sp + dpos_vp) * mask_pos) / prm.fpos
    cost_neg = T.sum((dneg + dneg_sp + dneg_vp) * mask_neg) / prm.fneg
    cost = cost_pos + cost_neg

    posdiff = T.sum(l1norm * T.eq(y_mat, 0) * mask_pos) / numpos
    negdiff = T.sum(l1norm * T.eq(y_mat, 1) * mask_neg) / numneg

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    updates = []
    for param_i, grad_i in zip(params, grads):
        updates.append((param_i, param_i - learning_rate * grad_i))

    cnn_model.x0 = x0
    cnn_model.x1 = x1
    cnn_model.p0 = p0
    cnn_model.p1 = p1
    # cnn_model.y  = y
    cnn_model.ys = ys
    cnn_model.yv = yv
    cnn_model.cost = cost
    cnn_model.cost_pos = cost_pos
    cnn_model.cost_neg = cost_neg

    cnn_model.dpos = T.sum(dpos * mask_pos)
    cnn_model.dneg = T.sum(dneg * mask_neg)
    cnn_model.dpos_sp = T.sum(dpos_sp * mask_pos)
    cnn_model.dneg_sp = T.sum(dneg_sp * mask_neg)
    cnn_model.dpos_vp = T.sum(dpos_vp * mask_pos)
    cnn_model.dneg_vp = T.sum(dneg_vp * mask_neg)

    cnn_model.numpos = numpos
    cnn_model.numneg = numneg
    cnn_model.l1 = l1norm
    cnn_model.posdiff = posdiff
    cnn_model.negdiff = negdiff

    cnn_model.grads = grads
    cnn_model.updates = updates

    cnn_model.update_params = params

    cnn_model.sketch_code = sketch_code
    cnn_model.sketch_output = sketch_output
    cnn_model.view_code = view_code
    cnn_model.view_output = view_output

    cnn_model.sketch_peer_output = sketch_peer_output
    cnn_model.view_peer_output = view_peer_output


    return cnn_model


def gen_train_func(datasetSh, trainSketchViewInds, train_labels, model):

    batch_size = model.setting.batch_size
    index = T.lscalar()
    sketches, views = datasetSh

    train_func_fast = theano.function([index], [model.cost, model.dpos, model.dneg, \
                                           model.dpos_sp, model.dneg_sp, \
                                           model.dpos_vp, model.dneg_vp], updates=model.updates,
          givens={
            model.x0: sketches[trainSketchViewInds[index * batch_size: (index + 1) * batch_size, 0]],
            model.x1: views[trainSketchViewInds[index * batch_size: (index + 1) * batch_size, 1]],
            model.p0: sketches[trainSketchViewInds[index * batch_size: (index + 1) * batch_size, 2]],
            model.p1: views[trainSketchViewInds[index * batch_size: (index + 1) * batch_size, 3]],
            model.ys: train_labels[index * batch_size: (index + 1) * batch_size, 1],
            model.yv: train_labels[index * batch_size: (index + 1) * batch_size, 2] })

    train_func = theano.function([index], [model.cost, model.dpos, model.dneg, \
                                           model.dpos_sp, model.dneg_sp, \
                                           model.dpos_vp, model.dneg_vp, \
                                           model.numpos, model.numneg, model.l1, model.posdiff, model.negdiff] + model.grads, updates=model.updates,
          givens={
            model.x0: sketches[trainSketchViewInds[index * batch_size: (index + 1) * batch_size, 0]],
            model.x1: views[trainSketchViewInds[index * batch_size: (index + 1) * batch_size, 1]],
            model.p0: sketches[trainSketchViewInds[index * batch_size: (index + 1) * batch_size, 2]],
            model.p1: views[trainSketchViewInds[index * batch_size: (index + 1) * batch_size, 3]],
            model.ys: train_labels[index * batch_size: (index + 1) * batch_size, 1],
            model.yv: train_labels[index * batch_size: (index + 1) * batch_size, 2] })

    apply_func = theano.function([model.x0, model.x1, model.p0, model.p1, model.ys, model.yv], [model.sketch_code, model.view_code, model.cost])
    apply_cnn = theano.function([model.x0], [model.sketch_code])

    return train_func_fast, train_func, apply_func, apply_cnn


def init_model(cnn_setting, params_val, prm):

    inputWH = prm.inputWH

    if prm.model_type == 'dual':
        model = gen_dual_models(prm, cnn_setting, inputWH, params_init_val=params_val)
    else:
        model = gen_models(prm, cnn_setting[0], inputWH, params_init_val=params_val)

    return model


