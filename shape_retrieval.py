import os, sys
import copy
import ConfigParser
import numpy
import cPickle

import time
import theano
import theano.tensor as T
import scipy.io as sio
import scipy.ndimage as ndimage

from cnn_model import *
from pprint import pprint


cnn_setting = [None, None]

class MyParameters(object):
    """all parameters"""
    def __init__(self):
        self.batch_size = 50
        self.n_epochs = 20
        self.lr_fc = 1.0
        self.lr_ff = 1.0
        self.learning_rate = 0.0001
        self.code_len = 64
        self.inputWH = 100
        self.exp_name = 'shrec'
        self.exp_suffix = 'debug'
        self.vp_num = 2
        self.render_type = 'comp'

        self.use_dropout = True
        self.data_aug = True

        self.model_type = 'dual'
        self.dataset = 'shrec'

        self.restart_epoch = -1
        self.disp = 0

        self.Cpos = 0.2
        self.Cneg = 10.0

        self.ppos = 0.5
        self.pneg = 0.06
        self.fpos = 1.0
        self.fneg = 2.0

        self.test_mode = 'test'

        self.cache_dir = './cache'
        self.data_dir = './data'

def padData(datax, batch_size):

    oriDataNum = datax.shape[0]
    if oriDataNum % batch_size != 0:
        padNum = (oriDataNum // batch_size + 1) * batch_size
        padInd = range(oriDataNum)
        padInd.extend(range(padNum-oriDataNum))
        datax = datax[padInd]

    return datax

def padBatch(sketchViewIndsLabels, batch_size):
    sketchViewInds, labels = sketchViewIndsLabels
    oriLen = len(labels)
    if oriLen % batch_size != 0:
        newLen = (oriLen // batch_size + 1) * batch_size
        extInd = range(oriLen)
        extInd.extend(range(newLen-oriLen))
        sketchViewInds = sketchViewInds[extInd]
        labels = labels[extInd]

    return sketchViewInds, labels


def load_data(prm, mode='train'):

    if mode == 'train':
        view_file = prm.view_train
        view_data = sio.loadmat(view_file)
        views = view_data['datax']
        view_label = view_data['datay'].astype('int32')

        sketch_file = prm.sketch_train
        sketch_data = sio.loadmat(sketch_file)
        sketches = sketch_data['datax']
        sketch_label = sketch_data['datay'].astype('int32')
        if prm.data_aug:
            print "add augmentation data ..."
            data_aug1 = sio.loadmat(prm.sketch_warp1)
            data_aug2 = sio.loadmat(prm.sketch_warp2)

            sketches = numpy.vstack((sketches, data_aug1['datax'], data_aug2['datax']))
            sketch_label = numpy.vstack((sketch_label, sketch_label, sketch_label))

        return sketches, views, sketch_label, view_label


    elif mode == 'test':
        view_file = prm.view_test
        view_data = sio.loadmat(view_file)
        views = view_data['datax']
        view_label = view_data['datay'].astype('int32')

        sketch_file = prm.sketch_test
        sketch_data = sio.loadmat(sketch_file)
        sketches = sketch_data['datax']
        sketch_label = sketch_data['datay'].astype('int32')

        return sketches, views, sketch_label, view_label

    else:
        assert False, 'unknown shrec data loading mode: %s' % (mode)


def load_pairs(prm):

    pair_data = sio.loadmat(prm.pair_file)
    triples = pair_data['triples']
    labels = pair_data['labels']

    return triples, labels

def get_random_pairs2(prm, sketches_label, views_label):

    triples = []
    labels = []
    sketch_pairs = numpy.zeros((0,2))

    sketches_label = numpy.asarray(sketches_label)
    views_label = numpy.asarray(views_label)
    # 2 views for each 3D model
    # 90 classes for training
    categories = numpy.unique(sketches_label)

    sample_pool = numpy.arange(views_label.shape[0])

    max_sketch_per_class = 50
    max_pos_per_sample = 50
    max_neg_per_sample = 50

    posCt = 0
    negCt = 0

    for c in categories.tolist():
        inclass_sketch = numpy.where(sketches_label==c)[0]
        sketch_ind = numpy.arange(len(inclass_sketch))
        prm.prng.shuffle(sketch_ind)
        inclass_sketch = inclass_sketch[sketch_ind]
        selected_sketch = inclass_sketch[sketch_ind[0:min(max_sketch_per_class, len(sketch_ind))]]

        other_sketch    = numpy.where(sketches_label!=c)[0]

        selected_view = numpy.where(views_label==c)[0]
        view_ind = numpy.arange(len(selected_view))
        prm.prng.shuffle(view_ind)
        view_ind = view_ind[0:min(max_pos_per_sample, len(view_ind))]
        selected_view = selected_view[view_ind]

        other_view    = numpy.where(views_label!=c)[0]
        view_ind = numpy.arange(len(other_view))
        prm.prng.shuffle(view_ind)
        view_ind = view_ind[0:min(max_neg_per_sample, len(view_ind))]
        other_view = other_view[view_ind]


        sketch_pos_ind = prm.prng.randint(len(inclass_sketch), size=(len(selected_view), 1))
        selected_sketch_peer = inclass_sketch[sketch_pos_ind]
        view_pos_ind = numpy.arange(len(selected_view))
        prm.prng.shuffle(view_pos_ind)
        selected_view_peer = selected_view[view_pos_ind]

        sketch_neg_ind = prm.prng.randint(len(other_sketch), size=(len(other_view), 1))
        other_sketch_peer = other_sketch[sketch_neg_ind]
        view_neg_ind = prm.prng.randint(len(selected_view), size=(len(other_view), 1))
        other_view_peer = selected_view[view_neg_ind]

        for sketchId in selected_sketch:
            for i in xrange(len(selected_view)):
                viewId = selected_view[i]
                sketchPeerId = selected_sketch_peer[i]
                viewPeerId = selected_view_peer[i]

                triples.append((sketchId, viewId, sketchPeerId, viewPeerId))
                labels.append((0,sketches_label[sketchId], views_label[viewId]))
                posCt += 1

            for i in xrange(len(other_view)):
                viewId = other_view[i]
                sketchPeerId = other_sketch_peer[i]
                viewPeerId   = other_view_peer[i]

                triples.append((sketchId, viewId, sketchPeerId, viewPeerId))
                labels.append((1,sketches_label[sketchId], views_label[viewId]))
                negCt += 1


    print 'sample pool size = %d, number of triples = %d, #pos=%d, #neg=%d' % (len(sample_pool), len(triples), posCt, negCt)

    triples = numpy.asarray(triples)
    labels = numpy.asarray(labels)

    validInd = numpy.arange(triples.shape[0])
    prm.prng.shuffle(validInd)
    triples = triples[validInd]
    labels = labels[validInd]

    return triples, sketch_pairs, labels


def calc_distmat(sketch_feat, view_feat):
    block_size = 2
    sketch_num = sketch_feat.shape[0]
    view_num = view_feat.shape[0]
    dist_mat = numpy.zeros((sketch_num, view_num))

    view_feat_mat = numpy.tile(view_feat, (block_size, 1, 1)).astype('float32')
    for i in xrange(sketch_num//block_size):
        tmp = numpy.abs(sketch_feat[i*block_size : (i+1) * block_size, numpy.newaxis, :]-view_feat_mat)
        dist_mat[i*block_size : (i+1) * block_size] = numpy.sum(tmp, axis=2)

    return dist_mat


def train(datasetSh, metaDataSh, prm, model, params_val=None, disp=False):

    sketches, views = datasetSh
    trainSketchViewInds, train_labels = metaDataSh

    train_ind = trainSketchViewInds.get_value()
    train_label = train_labels.get_value()

    pair_label = train_labels.get_value()

    batch_size = prm.batch_size
    inputWH = prm.inputWH
    n_epochs = prm.n_epochs
    learning_rate = prm.learning_rate
    code_len = prm.code_len


    n_train_samples = trainSketchViewInds.get_value(borrow=True).shape[0]
    n_train_batches = n_train_samples / batch_size

    print "train batch size:", batch_size, \
            " train batch number:", n_train_batches

    train_func_fast, train_func, apply_func, apply_cnn = \
            gen_train_func(datasetSh, trainSketchViewInds, train_labels, model)

    params = model.update_params

    print '... training'
    best_params = None
    best_average_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = prm.restart_epoch

    done_looping = False
    params_val = [None] * len(params)

    all_cost = numpy.zeros((n_train_batches, ))
    dpos = 0; dneg = 0;
    dpos_sp = 0; dneg_sp = 0;
    dpos_vp = 0; dneg_vp = 0;

    diff = numpy.zeros((n_train_samples, ))
    numpos = numpy.zeros((n_train_batches, ))
    numneg = numpy.zeros((n_train_batches, ))

    print '     #iter     |   cost    |     dpos   |   dneg   |   dpos_sp  | dneg_sp  |   dpos_vp  | dneg_vp  |   diff   |  posdiff  |  negdiff  | #pos | #neg '

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1

        # shuffle pairs
        idx = numpy.arange(train_ind.shape[0])
        prm.prng.shuffle(idx)
        train_ind = train_ind[idx]
        train_label = train_label[idx]

        metaDataSh[0].set_value(train_ind.astype('int32'))
        metaDataSh[1].set_value(train_label.astype('int32'))

        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 20 == 0:
                temp = train_func(minibatch_index)

                cost_ij = temp[0]
                grads = temp[12:]
                grad_vec = [numpy.mean(numpy.asarray(grad)) for grad in grads]
                all_cost[minibatch_index] = temp[0]

                dpos += temp[1];  dneg += temp[2]
                dpos_sp += temp[3];  dneg_sp += temp[4]
                dpos_vp += temp[5];  dneg_vp += temp[6]

                numpos[minibatch_index] = temp[7]
                numneg[minibatch_index] = temp[8]

            else:
                temp = train_func_fast(minibatch_index)
                all_cost[minibatch_index] = temp[0]
                dpos += temp[1];  dneg += temp[2]
                dpos_sp += temp[3];  dneg_sp += temp[4]
                dpos_vp += temp[5];  dneg_vp += temp[6]


            if iter % 20 == 0:
                batch_label = pair_label[minibatch_index*batch_size : (minibatch_index+1)*batch_size, :]
                npos = numpos[minibatch_index]
                nneg = numneg[minibatch_index]

                print   '  %8d | %6.6f |c  %2.6f | %2.6f |s  %2.6f | %2.6f |v  %2.6f | %2.6f | %2.6f |p %2.6f |n %2.6f | %4d | %4d ' % \
                      (iter, cost_ij, temp[1]/npos, temp[2]/nneg, \
                       temp[3]/npos, temp[4]/nneg, \
                       temp[5]/npos, temp[6]/nneg, \
                       numpy.mean(temp[9]), temp[10], temp[11], npos, nneg)


                p_mean = numpy.zeros((len(params),))
                p_std = numpy.zeros((len(params),))
                for k in xrange(len(params)):
                    tmp = params[k].get_value()
                    p_mean[k] = numpy.mean(tmp)
                    p_std[k] = numpy.std(tmp)

                if iter % 1000 == 0:
                    print 'parameter mean:', p_mean
                    print 'parameter std:', p_std


            if minibatch_index == n_train_batches-1  or minibatch_index % 5000 == 4999:

                this_average_loss = numpy.mean(all_cost)
                print('epoch %i, minibatch %i/%i, training error %f' % \
                      (epoch, minibatch_index + 1, n_train_batches, \
                      this_average_loss))

                total_pos = numpy.sum(numpos[0:minibatch_index])
                total_neg = numpy.sum(numneg[0:minibatch_index])
                print '    average dpos=%f, dneg=%f, dpos_sp=%f, dneg_sp=%f, dpos_vp=%f, dneg_vp=%f, number of pos samples=%d,  neg samples=%d' % (dpos/total_pos, \
                            dneg/total_neg, dpos_sp/total_pos, dneg_sp/total_neg, dpos_vp/total_pos, dneg_vp/total_neg, total_pos, total_neg)

                params_val = [ model.params[k].get_value() \
                              for k in xrange(len(model.params)) ]

                if minibatch_index == n_train_batches-1:
                    save_filename = prm.model_name+'-epoch%d.pkl' % epoch
                else:
                    save_filename = prm.model_name+'-epoch%d-iter%d.pkl' % (epoch, minibatch_index)

                f = file(save_filename, 'wb')
                cPickle.dump((best_average_loss, params_val, model.setting), f, protocol=cPickle.HIGHEST_PROTOCOL )
                f.close()

                # if we got the best validation score until now
                if this_average_loss < best_average_loss:
                    best_average_loss = this_average_loss
                    best_iter = iter

        timeSofar = (time.clock() - start_time) / 60.0
        print 'training time elapsed %.1f min' % timeSofar


    end_time = time.clock()
    print('Optimization complete.')
    print('Best average cost is %f  obtained at iteration %i' %
          (best_average_loss, best_iter + 1))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    run_epoch = epoch - prm.restart_epoch
    return (best_average_loss, params_val, run_epoch)

def train_dual_model(prm):

    global cnn_setting
    print 'Training using SHREC dataset ... '

    restart_epoch = prm.restart_epoch

    disp = prm.disp
    batch_size = prm.batch_size
    vp_num = prm.vp_num

    sketches, views, sketch_label, view_label = load_data(prm)
    oriSketchNum = sketches.shape[0]
    oriViewNum   = views.shape[0]

    sketchPad = padData(sketches, batch_size)
    viewPad = padData(views, batch_size)

    datasetSh = [theano.shared(sketchPad.astype('float32')), \
                 theano.shared(viewPad.astype('float32'))]

    sketchPadNum = sketchPad.shape[0]
    viewPadNum = viewPad.shape[0]
    n_sketch_batches = sketchPadNum // batch_size
    n_view_batches = viewPadNum // batch_size

    prm.inputWH = int(numpy.sqrt(sketches.shape[1]))
    print '\ninput patch size: (%d, %d)' % (prm.inputWH, prm.inputWH)

    params_val = None
    prm.restart_epoch = 0

    if restart_epoch >= 0:
        prm.restart_epoch = restart_epoch
        model_file = prm.model_name + '-epoch%d.pkl'%(restart_epoch)
        print "loading model file", model_file
        f = file(model_file, 'rb')
        model_data = cPickle.load(f)
        f.close()
        params_val = model_data[1]
        print '   load model parameters from ', model_file
        prm.restart_epoch = restart_epoch

    model = init_model(cnn_setting, params_val, prm)

    train_ind, train_label = load_pairs(prm)

    # idx = numpy.arange(train_ind.shape[0])
    # prm.prng.shuffle(idx)
    # train_ind = train_ind[idx]
    # train_label = train_label[idx]


    total_num = train_label.shape[0]
    neg_num = numpy.sum(train_label[:,0])
    print 'training pairs %d (%d pos, %d neg)' % (total_num, total_num-neg_num, neg_num)


    oriTrainNum = len(train_ind)
    prm.oriTrainNum = oriTrainNum

    trainSketchViewIndsLabels = padBatch([train_ind, train_label], prm.batch_size)

    print 'original training triples %d, padded to %d' % \
            (oriTrainNum, len(trainSketchViewIndsLabels[0]))

    trainSketchViewInds, train_labels = trainSketchViewIndsLabels
    metaDataSh =  ( theano.shared(trainSketchViewInds.astype('int32'), borrow=True), \
                    theano.shared(train_labels.astype('int32'), borrow=True) )

    print 'sample number of train %d' % \
            ( len(trainSketchViewIndsLabels[0]) )

    print 'CNN training...'

    average_loss, params_val, run_epoch = train( \
                            datasetSh, metaDataSh, prm, model, params_val=params_val, \
                            disp=disp)

    print 'average loss:', average_loss*100, '%'


def test_dual_model(prm, epoch_list):

    sketches, views, sketch_label, view_label = load_data(prm, mode=prm.test_mode)

    for epoch in epoch_list:
        model_file = prm.model_name + '-epoch%d.pkl'% (epoch )
        feature_file = prm.feat_prefix + '-epoch%d-%s.mat' % (epoch, prm.test_mode)
        if not os.path.isfile(model_file):
            print 'model file does not exist, skip testing:', model_file
            continue
        if os.path.isfile(feature_file):
            print 'feature file already exist, skip testing:', feature_file
            continue

        print 'model file:', model_file
        print 'feature file:', feature_file

        sketch_feats, view_feats, dist_mat = do_retrieval(prm, sketches, views, model_file)

        print 'saving results ...'
        feat_data = {'sketch_feat': sketch_feats, 'view_feat': view_feats, 'dist_mat': dist_mat,
                     'sketch_label': sketch_label, 'view_label': view_label }
        sio.savemat(feature_file, feat_data, oned_as='column')


def do_retrieval(prm, sketches, views, model_file):

    global cnn_setting

    print 'load model parameters', model_file
    f = file(model_file, 'rb')
    model_data = cPickle.load(f)
    params_val  = model_data[1]
    f.close()

    print 'setup model and functions ...'

    disp = prm.disp

    batch_size = prm.batch_size

    prm.inputWH = int(numpy.sqrt(sketches.shape[1]))
    print '\ninput patch size: (%d, %d) ' % (prm.inputWH, prm.inputWH)

    # prm.code_len = params_val[6].shape[1]
    print 'load model parameters, code length is :', prm.code_len

    oriSketchesNum = sketches.shape[0]
    oriViewNum = views.shape[0]
    sketches = padData(sketches, batch_size)
    views = padData(views, batch_size)

    print 'do_retrieval() : sketches.shape=', sketches.shape

    model = init_model(cnn_setting, params_val, prm)

    sketchesSh = theano.shared(sketches.astype('float32'), borrow=True)
    viewsSh = theano.shared(views.astype('float32'), borrow=True)

    index = T.lscalar()

    sketch_calc_func = theano.function([index], model.sketch_code,
          givens={
            model.x0: sketchesSh[index * batch_size: (index + 1) * batch_size] })

    view_calc_func = theano.function([index], model.view_code,
          givens={
            model.x1: viewsSh[index * batch_size: (index + 1) * batch_size] })

    apply_cnn = theano.function([model.x0], model.sketch_output)

    n_sketch_batches = sketches.shape[0]//batch_size
    n_view_batches = views.shape[0]//batch_size

    rotate_angle = [0]

    sketch_feats = numpy.zeros((sketches.shape[0], prm.code_len))
    view_feat = numpy.zeros((views.shape[0], prm.code_len))
    view_feats = numpy.zeros((views.shape[0], prm.code_len))

    print 'calculate view features ...'
    for i in xrange(n_view_batches):
        view_feat[i*batch_size : (i+1) * batch_size] = view_calc_func(i)

    view_feats = view_feat[0:oriViewNum]

    start_time = time.clock()
    print 'calculate sketch features ...'
    for i in xrange(n_sketch_batches):
        sketch_feats[i*batch_size : (i+1) * batch_size] = sketch_calc_func(i)

    sketch_feats = sketch_feats[0:oriSketchesNum]
    dist_mat = calc_distmat(sketch_feats, view_feats)

    time_elapsed = time.clock() - start_time
    print 'sketch processing time %.3f sec' % time_elapsed

    return sketch_feats, view_feats, dist_mat


def parseConfig(filename):
    config = ConfigParser.ConfigParser()
    config.read(filename)
    return config


def setupExperiment(config, case, runMode="Train"):
    # global experiment settings
    prm = MyParameters()

    assert config.has_option(case, 'name'), 'TestCase(): name must be set'
    assert config.has_option(case, 'cnn_spec_ss'), 'TestCase(): cnn_spec_ss must be set'
    assert config.has_option(case, 'cnn_spec_vs'), 'TestCase(): cnn_spec_vs must be set'

    prm.exp_name = config.get(case, 'name')
    prm.exp_suffix = case

    model_dir = '%s/model-%s-%s' % (prm.cache_dir, prm.exp_name, prm.exp_suffix)
    feats_dir = '%s/feats-%s-%s' % (prm.cache_dir, prm.exp_name, prm.exp_suffix)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(feats_dir):
        os.makedirs(feats_dir)

    prm.model_dir = model_dir
    prm.feats_dir = feats_dir
    prm.model_name = '%s/%s-%s' % (model_dir, prm.exp_name, prm.exp_suffix)
    prm.feat_prefix = '%s/%s-%s' % (feats_dir, prm.exp_name, prm.exp_suffix)

    if config.has_option(case, 'epoch_list'):
        epoch_list_str = config.get(case, 'epoch_list')
        prm.epoch_list = [int(epoch) for epoch in epoch_list_str.split()]
    if config.has_option(case, 'n_epochs'):
        prm.n_epochs= config.getint(case, 'n_epochs')
    if config.has_option(case, 'batch_size'):
        prm.batch_size = config.getint(case, 'batch_size')
    if config.has_option(case, 'vp_num'):
        prm.vp_num = config.getint(case, 'vp_num')
    if config.has_option(case, 'ds'):
        prm.inputWH = config.getint(case, 'ds')
    if config.has_option(case, 'code_len'):
        prm.code_len = config.getint(case, 'code_len')
    if config.has_option(case, 'disp'):
        prm.disp = config.getint(case, 'disp')

    if config.has_option(case, 'model_type'):
        prm.model_type = config.get(case, 'model_type')
    if config.has_option(case, 'lr_fc'):
        prm.lr_fc = config.getfloat(case, 'lr_fc')
    if config.has_option(case, 'lr_ff'):
        prm.lr_ff = config.getfloat(case, 'lr_ff')
    if config.has_option(case, 'learning_rate'):
        prm.learning_rate = config.getfloat(case, 'learning_rate')
    if config.has_option(case, 'restart_epoch'):
        prm.restart_epoch = config.getint(case, 'restart_epoch')
    if config.has_option(case, 'test_mode'):
        prm.test_mode = config.get(case, 'test_mode')
    if config.has_option(case, 'dataset'):
        prm.dataset = config.get(case, 'dataset')
    if config.has_option(case, 'pair_file'):
        prm.pair_file = prm.data_dir + '/' + config.get(case, 'pair_file')
    if config.has_option(case, 'data_aug'):
        prm.data_aug = config.get(case, 'data_aug')

    if prm.dataset == 'shrec':
        prm.sketch_train = prm.data_dir + '/sketch_train.mat'
        prm.sketch_test = prm.data_dir + '/sketch_test.mat'
        prm.view_train = prm.data_dir + '/view_shrec.mat'
        prm.view_test = prm.data_dir + '/view_shrec.mat'
        prm.sketch_warp1 = prm.data_dir + '/sketches_train_augment_warp.mat'
        prm.sketch_warp2 = prm.data_dir + '/sketches_train_augment_warp2.mat'

    elif prm.dataset == 'shrec14':
        prm.sketch_train = prm.data_dir + '/shrec14_sketch_train.mat'
        prm.sketch_test = prm.data_dir + '/shrec14_sketch_test.mat'
        prm.view_train = prm.data_dir + '/shrec14_view_all.mat'
        prm.view_test = prm.data_dir + '/shrec14_view_all.mat'
        prm.sketch_warp1 = prm.data_dir + '/shrec14_sketch_train_augment_warp.mat'
        prm.sketch_warp2 = prm.data_dir + '/shrec14_sketch_train_augment_warp2.mat'
        prm.pair_file = prm.data_dir + '/shrec14_train_pairs.mat'

    elif prm.dataset == 'psb':
        prm.sketch_train = prm.data_dir + '/psb_sketch_train.mat'
        prm.sketch_test = prm.data_dir + '/psb_sketch_test.mat'
        prm.view_train = prm.data_dir + '/psb_view_train.mat'
        prm.view_test = prm.data_dir + '/psb_view_test.mat'
        prm.sketch_warp1 = prm.data_dir + '/psb_sketches_train_augment_warp.mat'
        prm.sketch_warp2 = prm.data_dir + '/psb_sketches_train_augment_warp2.mat'
        prm.pair_file = prm.data_dir + '/psb_train_pairs.mat'

    else:
        assert False, 'Unknown dataset setting %s' % prm.dataset

    prm.cnn_spec_ss = config.get(case, 'cnn_spec_ss')
    prm.cnn_spec_vs = config.get(case, 'cnn_spec_vs')
    if config.has_option(case, 'use_dropout'):
        prm.use_dropout = config.get(case, 'use_dropout')

    if config.has_option(case, 'Cpos'):
        prm.Cpos= config.getfloat(case, 'Cpos')
    if config.has_option(case, 'Cneg'):
        prm.Cneg = config.getfloat(case, 'Cneg')

    if config.has_option(case, 'ppos'):
        prm.ppos = config.getfloat(case, 'ppos')
    if config.has_option(case, 'pneg'):
        prm.pneg = config.getfloat(case, 'pneg')
    if config.has_option(case, 'fpos'):
        prm.fpos = config.getfloat(case, 'fpos')
    if config.has_option(case, 'fneg'):
        prm.fneg = config.getfloat(case, 'fneg')

    cnn_setting_ss = CNNSetting(prm.cnn_spec_ss)
    cnn_setting_ss.batch_size = prm.batch_size
    cnn_setting_ss.nout = prm.code_len
    cnn_setting_ss.learning_rate = prm.learning_rate
    cnn_setting_ss.Cpos = prm.Cpos
    cnn_setting_ss.Cneg = prm.Cneg
    if prm.use_dropout:
        cnn_setting_ss.use_dropout = True


    cnn_setting_vs = CNNSetting(prm.cnn_spec_vs)
    cnn_setting_vs.batch_size = prm.batch_size
    cnn_setting_vs.nout = prm.code_len
    cnn_setting_vs.learning_rate = prm.learning_rate
    cnn_setting_vs.Cpos = prm.Cpos
    cnn_setting_vs.Cneg = prm.Cneg
    if prm.use_dropout:
        cnn_setting_vs.use_dropout = True

    print 'sketch structure setting: %s' % (prm.cnn_spec_ss)
    print 'view structure setting: %s' % (prm.cnn_spec_vs)

    cnn_setting = [cnn_setting_ss, cnn_setting_vs]

    # numpy.random.seed(prm.randSeed)
    prm.prng = numpy.random.RandomState()
    pprint(vars(prm))
    print ''

    return prm, cnn_setting

def main(argv):

    usage = '\nUsage: ' + argv[0] + \
            ' [train/test] [arguments]\n' + \
            'train [train_config_file]\n' + \
            'test  [test_config_file]\n'

    global cnn_setting

    runMode = None
    if len(argv) < 2:
        print usage
        return
    else:
        runMode = argv[1]

    if runMode == 'train':
        print 'Running mode : training'

        config_file = 'exp-shrec.cfg'
        if len(argv) > 2:
            config_file = argv[2]
        train_config = parseConfig(config_file)

        for train_case in train_config.sections():
            prm, cnn_setting = setupExperiment(train_config, train_case, runMode)
            train_dual_model(prm)

    elif runMode == 'test':
        print 'Running mode : testing'

        config_file = 'exp-shrec.cfg'
        if len(argv) > 2:
            config_file = argv[2]
        test_config = parseConfig(config_file)

        for test_case in test_config.sections():
            prm, cnn_setting = setupExperiment(test_config, test_case, runMode)
            # use raw data in testing
            prm.data_aug = False
            test_dual_model(prm, prm.epoch_list)

    else:
        print 'Unknown running mode'
        print usage
        return

if __name__ == '__main__':
    main(sys.argv)


