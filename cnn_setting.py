import numpy as np
from pprint import pprint, pformat


class CNNSetting(object):

    def __init__(self, spec=None):
        # input and output
        self.batch_size = 100
        self.nchannel = 1
        self.use_dropout = False
        self.dropout_p = 0.5
        self.nout = 64
        self.learning_rate = 0.0001
        self.final_active_func = 'relu'
        self.spec = spec
        self.Cpos = 0.2
        self.Cneg = 10.0
        self.alpha = 2.77
        self.sw = 1.0
        self.vw = 1.0

        # cnn structure
        self.init_ds100()

    def init_ds100(self):
        print 'initialize cnn setting: ds100'
        self.n_layer = 4
        self.tag = ['c', 'c', 'c', 'f']
        self.fs = [(13, 13), (7, 7), (3, 3), (3, 3)]
        self.ps = [(4, 4), (2, 2), (2, 2), (1, 1)]
        self.nkerns = [32, 64, 256]


    def __str__(self):

        msg = pformat(vars(self))
        msg = msg[1:len(msg)-1] + '\n'

        min_size = self.min_patch_size()
        msg += '\n Minimum accepted patch size : ' + min_size.__str__()
        msg += '\n Feature map size of each layer: ' + self.output_map_size(min_size).__str__()
        msg += '\n Neuron numbers: ' + self.neuron_num().__str__()
        msg += '\n Parameter numbers: ' + self.parameter_num().__str__()
        return msg

    def min_patch_size(self):
        ptsize0 = 1
        ptsize1 = 1
        ps = self.ps
        fs = self.fs

        for i in xrange(len(fs)-1, -1, -1):
            ptsize0 = (ptsize0*ps[i][0])+fs[i][0]-1
            ptsize1 = (ptsize1*ps[i][1])+fs[i][1]-1

        return (ptsize0, ptsize1)

    def get_sp_step(self):
        ps = [self.ps[i][0] for i in xrange(len(self.ps))]
        return np.prod(ps)

    def output_map_size(self, ptsize):
        fs = self.fs
        ps = self.ps

        out_list = [ptsize]
        out0 = ptsize[0]
        out1 = ptsize[1]

        for i in xrange(len(fs)):
            out0 = (out0-fs[i][0]+1)/ps[i][0]
            out1 = (out1-fs[i][1]+1)/ps[i][1]
            out_list.append((out0, out1))

        return out_list

    def neuron_num(self):
        fs = self.fs
        ps = self.ps
        ptsize = self.min_patch_size()

        out_list = [np.prod(ptsize) * self.nchannel]
        out0 = ptsize[0]
        out1 = ptsize[1]

        for i in xrange(len(fs)-1):
            out0 = (out0-fs[i][0]+1)/ps[i][0]
            out1 = (out1-fs[i][1]+1)/ps[i][1]
            out_list.append(out0 * out1 * self.nkerns[i])

        out_list.append(self.nout)

        return out_list

    def parameter_num(self):
        fs = self.fs
        ps = self.ps
        ptsize = self.min_patch_size()
        nkerns = self.nkerns[:]
        nkerns.append(self.nout)

        out_list = []
        outmapsz = self.output_map_size(self.min_patch_size())

        for i in xrange(len(fs)):
            para = (fs[i][0] * fs[i][1]) * nkerns[i]
            if i > 0:
                para *= nkerns[i-1]
            para += nkerns[i]

            conn = para * (outmapsz[i][0]-fs[i][0]+1) * (outmapsz[i][1]-fs[i][1]+1)

            out_list.append((para, conn))

        return out_list


if __name__ == '__main__':

    m = CNNSetting('ds100')
    print m
    # print m.output_map_size((100, 100))

