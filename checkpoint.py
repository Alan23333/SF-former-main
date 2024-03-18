import os
import torch
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Checkpoint():
    def __init__(self, opt):
        self.opt = opt
        self.ok = True
        self.log = torch.Tensor()

        self.dir = opt.save

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.dir)
        _make_dir(self.dir + '/model')
        _make_dir(self.dir + '/results')

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.dir, is_best=is_best)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        label = 'SR on {}'.format(self.opt.dataset)
        fig = plt.figure()
        plt.title(label)
        plt.plot(
            axis,
            self.log.numpy(),
            label='Scale {}'.format(self.opt.scale)
        )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(self.dir, self.opt.dataset))
        plt.close(fig)

    def save_results_nopostfix(self, filename, sr, scale):
        apath = '{}/results/{}/x{}'.format(self.dir, self.opt.dataset, scale)
        if not os.path.exists(apath):
            os.makedirs(apath)
        filename = os.path.join(apath, filename)

        normalized = sr * (255 / self.opt.data_range)
        ndarr = normalized
        sio.savemat(filename, {'sr': ndarr})