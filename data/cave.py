import os
from data import common
import numpy as np
import scipy.io as sio
import random
from torch.utils.data import Dataset


class TrainSet(Dataset):
    def __init__(self, args):
        self.args = args
        data = sio.loadmat('data/CAVE/response coefficient')
        self.R = data['R']
        self.B = data['B1']

        self.PrepareDataAndiniValue(self.R, self.args.prepare)
        self._set_dataset_length()

    def __getitem__(self, idx):
        idx = self._get_index(idx)

        X, Y = self.all_train_data_in(idx)
        X, Y, Z = self.train_data_in(X, Y, self.B, self.args.patch_size, self.args.scale)  # x(96,96,29) Y(96,96,3) Z(12,12,29)
        D, A = common.Upsample(Y, Z, self.R.T, self.B, self.args.scale)

        CGT = np.dot(D.T, common.hyperConvert2D(X))
        CGT = common.reshape(CGT.T, (self.args.patch_size, self.args.patch_size, -1))

        A = common.np2Tensor(A, data_range=self.args.data_range)
        CGT = common.np2Tensor(CGT, data_range=self.args.data_range)

        return A, CGT  # CGT 标签


    def __len__(self):
        return self.dataset_length

    def _set_dataset_length(self):
        self.dataset_length = self.args.test_every * self.args.batch_size

    def _get_index(self, idx):
        return idx % 20

    # Prepare dataset for training
    def PrepareDataAndiniValue(self, R, prepare='Yes'):
        DataRoad = 'data/CAVE/train/'
        if prepare != 'No':
            print('Generating the training dataset in folder data/CAVE/train')
            # the index will become traning dataset
            Ind = [2, 31, 25, 6, 27, 15, 19, 14, 12, 28, 26, 29, 8, 13, 22, 7, 24, 30, 10, 23]

            common.mkdir(DataRoad + 'X/')
            common.mkdir(DataRoad + 'Y/')

            for root, dirs, files in os.walk('data/CAVE/complete_ms_data/'):
                for i in range(20):
                    print('processing ' + dirs[Ind[i] - 1])
                    X = common.readImofDir(
                        'data/CAVE/complete_ms_data/' + dirs[Ind[i] - 1] + '/' + dirs[Ind[i] - 1]
                    )
                    Y = np.tensordot(X, R, (2, 0))
                    sio.savemat(DataRoad + 'X/' + dirs[Ind[i] - 1] + '.mat', {'X': X})
                    sio.savemat(DataRoad + 'Y/' + dirs[Ind[i] - 1] + '.mat', {'Y': Y})
                break

        else:
            print('Using the prepared trainset and initial values in folder data/CAVE/train')

    def all_train_data_in(self, idx):
        for root, dirs, files in os.walk('data/CAVE/train/X/'):
            data = sio.loadmat("data/CAVE/train/X/" + files[idx])
            X = data['X']
            data = sio.loadmat("data/CAVE/train/Y/" + files[idx])
            Y = data['Y']

        return X, Y

    def train_data_in(self, X, Y, C, sizeI, sf, channel=29):
        batch_Z = np.zeros((sizeI // sf, sizeI // sf, channel), 'f')
        px = random.randint(0, 512 - sizeI)
        py = random.randint(0, 512 - sizeI)
        subX = X[px:px + sizeI:1, py:py + sizeI:1, :]
        subY = Y[px:px + sizeI:1, py:py + sizeI:1, :]

        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)

        # Random rotation
        for j in range(rotTimes):
            subX = np.rot90(subX)
            subY = np.rot90(subY)

        # Random vertical Flip
        for j in range(vFlip):
            subX = subX[:, ::-1, :]
            subY = subY[:, ::-1, :]

        # Random Horizontal Flip
        for j in range(hFlip):
            subX = subX[::-1, :, :]
            subY = subY[::-1, :, :]

        batch_X = subX
        batch_Y = subY

        for i in range(channel):
            subZ = np.real(np.fft.ifft2(np.fft.fft2(batch_X[:, :, i]) * C))
            subZ = subZ[0:sizeI:sf, 0:sizeI:sf]
            batch_Z[:, :, i] = subZ

        return batch_X, batch_Y, batch_Z


class TestSet(Dataset):
    def __init__(self, args):
        self.args = args
        self.pad = 512
        data = sio.loadmat('data/CAVE/response coefficient')
        self.R = data['R']
        self.B = data['B2']

        self.PrepareDataAndiniValue(self.R, self.B, self.args.scale, self.args.prepare)
        self._set_dataset_length()

    def __getitem__(self, idx):
        X, Y, Z, filename = self.all_test_data_in(idx)
        D, A = common.Upsample(Y, Z, self.R.T, self.B, self.args.scale)

        A = common.np2Tensor(
            A, data_range=self.args.data_range
        )

        return X, Y, Z, D, A, filename, self.pad


    def __len__(self):
        return self.dataset_length

    def _set_dataset_length(self):
        self.dataset_length = 12

    # Prepare dataset for testing
    def PrepareDataAndiniValue(self, R, C, sf, prepare='Yes', channel=29):
        DataRoad = 'data/CAVE/test/'
        if prepare != 'No':
            print('Generating the testing dataset in folder data/CAVE/test')
            # random index will become testing dataset
            Ind = [18, 17, 21, 3, 9, 4, 20, 5, 16, 32, 11, 1]

            common.mkdir(DataRoad + 'X/')
            common.mkdir(DataRoad + 'Y/')
            common.mkdir(DataRoad + 'Z/')

            for root, dirs, files in os.walk('data/CAVE/complete_ms_data/'):
                for i in range(12):
                    Z = np.zeros([512 // sf, 512 // sf, channel])
                    print('processing ' + dirs[Ind[i] - 1])
                    X = common.readImofDir(
                        'data/CAVE/complete_ms_data/' + dirs[Ind[i] - 1] + '/' + dirs[Ind[i] - 1]
                    )
                    Y = np.tensordot(X, R, (2, 0))
                    for j in range(channel):
                        subZ = np.real(np.fft.ifft2(np.fft.fft2(X[:, :, j]) * C))
                        subZ = subZ[0::sf, 0::sf]
                        Z[:, :, j] = subZ
                    sio.savemat(DataRoad + 'X/' + dirs[Ind[i] - 1] + '.mat', {'X': X})
                    sio.savemat(DataRoad + 'Y/' + dirs[Ind[i] - 1] + '.mat', {'Y': Y})
                    sio.savemat(DataRoad + 'Z/' + dirs[Ind[i] - 1] + '.mat', {'Z': Z})
                break

        else:
            print('Using the prepared testset and initial values in folder data/CAVE/test')

    def all_test_data_in(self, idx):
        for root, dirs, files in os.walk('data/CAVE/test/X/'):
            filename = files[idx]
            data = sio.loadmat('data/CAVE/test/X/' + filename)
            X = data['X']
            data = sio.loadmat('data/CAVE/test/Y/' + filename)
            Y = data['Y']
            data = sio.loadmat('data/CAVE/test/Z/' + filename)
            Z = data['Z']

        return X, Y, Z, filename