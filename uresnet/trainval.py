from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import time
import os
import sys
from uresnet.ops import GraphDataParallel
import uresnet.models as models
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


class trainval(object):
    def __init__(self, flags):
        self._flags = flags
        self.tspent = {}
        self.tspent_sum = {}

    def backward(self):
        total_loss = 0.0
        for loss in self._loss:
            total_loss += loss
        total_loss /= len(self._loss)
        self._loss = []  # Reset loss accumulator

        self._optimizer.zero_grad()  # Reset gradients accumulation
        total_loss.backward()
        self._optimizer.step()

    def save_state(self, iteration):
        tstart = time.time()
        filename = '%s-%d.ckpt' % (self._flags.WEIGHT_PREFIX, iteration)
        torch.save({
            'global_step': iteration,
            'state_dict': self._net.state_dict(),
            'optimizer': self._optimizer.state_dict()
        }, filename)
        self.tspent['save'] = time.time() - tstart

    def train_step(self, data_blob, epoch=None, batch_size=1):
        tstart = time.time()
        self._loss = []  # Initialize loss accumulator
        res_combined = self.forward(data_blob,
                                    epoch=epoch, batch_size=batch_size)
        # Run backward once for all the previous forward
        self.backward()
        self.tspent['train'] = time.time() - tstart
        self.tspent_sum['train'] += self.tspent['train']
        return res_combined

    def forward(self, data_blob, epoch=None, batch_size=1):
        """
        Run forward for
        flags.BATCH_SIZE / (flags.MINIBATCH_SIZE * len(flags.GPUS)) times
        """
        res_combined = {}
        for idx in range(len(data_blob['data'])):
            blob = {}

            for key in data_blob.keys():
                blob[key] = data_blob[key][idx]

            data  = blob['data'][idx]
            if 'label' in blob:
                label = blob['label'][idx]
            if 'weight' in blob:
                weight = blob['weight'][idx]
#            if (idx % 10000 ==0):
#	    	print('number of points = %s and number of points removed for low Thresh is %s, number of points removed above high thresh is %s' %
#                    (np.sum(data[:, 3]>=0), np.sum(data[:, 3] <= 10),np.sum(data[:,3]>=300)))
            N_Thresh = np.sum(np.logical_and(data[:, 3] >= 10 ,data[:,3]<=300))
            i_Thresh = 0
            thresh_data  = np.zeros( (N_Thresh,data.shape[1]),  dtype=data.dtype  )
            if 'label' in blob:
		    thresh_label  = np.zeros( (N_Thresh,label.shape[1]),  dtype=label.dtype  )
            if 'weight' in blob:
		    thresh_weight  = np.zeros( (N_Thresh,weight.shape[1]),  dtype=weight.dtype  )

            for i in range(data.shape[0]):
                if(np.logical_and(data[i,3]>=10 ,data[i,3]<=300)):
                    thresh_data[i_Thresh,:] =data[i,:]
                    if 'label' in blob:
                        thresh_label[i_Thresh,:] = label[i,:]
		    if 'weight' in blob:
                        thresh_weight[i_Thresh,:] = label[i,:]

                    i_Thresh +=1

            blob['data'][idx]  = thresh_data
            if 'label' in blob:
                blob['label'][idx]  = thresh_label
	    if 'weight' in blob:
                blob['weight'][idx]  = thresh_weight




            res = self._forward(blob,
                                epoch=epoch)
            for key in res.keys():
                if key not in res_combined:
                    res_combined[key] = res[key]
                else:
                    res_combined[key].extend(res[key])
        # Average loss and acc over all the events in this batch
        res_combined['accuracy'] = np.array(res_combined['accuracy']).sum() / batch_size
        res_combined['loss_seg'] = np.array(res_combined['loss_seg']).sum() / batch_size
        return res_combined

    def _forward(self, data_blob, epoch=None):
        """
        data/label/weight are lists of size minibatch size.
        For sparse uresnet:
        data[0]: shape=(N, 5)
        where N = total nb points in all events of the minibatch
        For dense uresnet:
        data[0]: shape=(minibatch size, channel, spatial size, spatial size, spatial size)
        """
        data = data_blob['data']
        label = data_blob.get('label', None)
        weight = data_blob.get('weight', None)
        # matplotlib.image.imsave('data0.png', data[0, 0, ...])
        # matplotlib.image.imsave('data1.png', data[1, 0, ...])
        # print(label.shape, np.unique(label, return_counts=True))
        # matplotlib.image.imsave('label0.png', label[0, 0, ...])
        # matplotlib.image.imsave('label1.png', label[1, 0, ...])
        with torch.set_grad_enabled(self._flags.TRAIN):
            # Segmentation
            data = [torch.as_tensor(d) for d in data]
            if torch.cuda.is_available():
                data = [d.cuda() for d in data]
            else:
                data = data[0]
            tstart = time.time()
            segmentation = self._net(data)
            if not torch.cuda.is_available():
                data = [data]

            # If label is given, compute the loss
            loss_seg, acc = 0., 0.
            if label is not None:
                label = [torch.as_tensor(l) for l in label]
                if torch.cuda.is_available():
                    label = [l.cuda() for l in label]
                # else:
                #     label = label[0]
                for l in label:
                    l.requires_grad = False
                # Weight is optional for loss
                if weight is not None:
                    weight = [torch.as_tensor(w) for w in weight]
                    if torch.cuda.is_available():
                        weight = [w.cuda() for w in weight]
                    # else:
                    #     weight = weight[0]
                    for w in weight:
                        w.requires_grad = False
                loss_seg, acc = self._criterion(segmentation, data, label, weight)
                if self._flags.TRAIN:
                    self._loss.append(loss_seg)
            res = {
                'segmentation': [s.cpu().detach().numpy() for s in segmentation],
                'softmax': [self._softmax(s).cpu().detach().numpy() for s in segmentation],
                'accuracy': [acc],
                'loss_seg': [loss_seg.cpu().item() if not isinstance(loss_seg, float) else loss_seg]
            }
            self.tspent['forward'] = time.time() - tstart
            self.tspent_sum['forward'] += self.tspent['forward']
            return res

    def initialize(self):
        # To use DataParallel all the inputs must be on devices[0] first
        model = None
        if self._flags.MODEL_NAME == 'uresnet_sparse':
            model = models.SparseUResNet
            self._criterion = models.SparseSegmentationLoss(self._flags)
        elif self._flags.MODEL_NAME == 'uresnet_dense':
            model = models.DenseUResNet
            self._criterion = models.DenseSegmentationLoss(self._flags)
        else:
            raise Exception("Unknown model name provided")

        self.tspent_sum['forward'] = self.tspent_sum['train'] = self.tspent_sum['save'] = 0.
        self.tspent['forward'] = self.tspent['train'] = self.tspent['save'] = 0.

        # if len(self._flags.GPUS) > 0:
        self._net = GraphDataParallel(model(self._flags),
                                      device_ids=self._flags.GPUS,
                                      dense=('sparse' not in self._flags.MODEL_NAME))
        # else:
        #     self._net = model

        if self._flags.TRAIN:
            self._net.train()
        else:
            self._net.eval()

        if torch.cuda.is_available():
            self._net.cuda()
            self._criterion.cuda()

        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self._flags.LEARNING_RATE)
        self._softmax = torch.nn.Softmax(dim=1 if 'sparse' in self._flags.MODEL_NAME else 0)

        iteration = 0
        if self._flags.MODEL_PATH:
            if not os.path.isfile(self._flags.MODEL_PATH):
                sys.stderr.write('File not found: %s\n' % self._flags.MODEL_PATH)
                raise ValueError
            print('Restoring weights from %s...' % self._flags.MODEL_PATH)
            with open(self._flags.MODEL_PATH, 'rb') as f:
                if len(self._flags.GPUS) > 0:
                    checkpoint = torch.load(f)
                else:
                    checkpoint = torch.load(f, map_location='cpu')
                # print(checkpoint['state_dict']['module.conv1.1.running_mean'],
                #       checkpoint['state_dict']['module.conv1.1.running_var'])
                # for key in checkpoint['state_dict']:
                #     if key not in self._net.state_dict():
                #         checkpoint['state_dict'].pop(key, None)
                #         print('Ignoring %s' % key)
                # new_state = self._net.state_dict()
                # new_state.update(checkpoint['state_dict'])
                self._net.load_state_dict(checkpoint['state_dict'], strict=False)
                if self._flags.TRAIN:
                    # This overwrites the learning rate, so reset the learning rate
                    self._optimizer.load_state_dict(checkpoint['optimizer'])
                    for g in self._optimizer.param_groups:
                        g['lr'] = self._flags.LEARNING_RATE
                iteration = checkpoint['global_step'] + 1
            print('Done.')

        return iteration
