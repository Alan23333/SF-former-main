import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from data import common
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset


class TestSet(Dataset):
    def __init__(self, args):
        self.args = args
        self.pad = 640  # to set the cropping size of the Harvard dataset
        data = sio.loadmat('data/UH/response coefficient')
        self.R = data['R']
        self.B = data['B']
        self.B = self.B[0:640, 0:640]

        self.PrepareDataAndiniValue(self.R, self.B, self.args.scale, self.args.prepare, pad=self.pad)
        self._set_dataset_length()

    def __getitem__(self, idx):
        Y, Z, filename = self.all_test_data_in(0)
        Y = Y[0:640,0:640,:]
        Z = Z[0:80, 0:80, :]

        D, A = common.Upsample(Y, Z, self.R.T, self.B, self.args.scale)

        A = common.np2Tensor(
            A, data_range=self.args.data_range
        )

        return Y, Z, D, A, filename, self.pad


    def __len__(self):
        return self.dataset_length

    def _set_dataset_length(self):
        self.dataset_length = 10

    # Prepare dataset for testing
    def PrepareDataAndiniValue(self, R, C, sf, prepare='Yes', channel=29, pad=1040):
        DataRoad = 'data/UH/test/'
        if prepare != 'No':
            print('Generating the testing dataset in folder data/UH/test')

            common.mkdir(DataRoad + 'X/')
            common.mkdir(DataRoad + 'Y/')
            common.mkdir(DataRoad + 'Z/')

            for root, dirs, files in os.walk('data/UH/complete_ms_data/'):
                for i in range(10):
                    Z = np.zeros([pad // sf, pad // sf, channel])
                    print('processing ' + str(i + 1))
                    data = sio.loadmat('data/UH/complete_ms_data/' + files[i])
                    X = data['ref']
                    X = X[0:pad, 0:pad, 2:31] / np.max(X) * 255
                    Y = np.tensordot(X, R, (2, 0))
                    for j in range(channel):
                        subZ = np.real(np.fft.ifft2(np.fft.fft2(X[:, :, j]) * C))
                        subZ = subZ[0::sf, 0::sf]
                        Z[:, :, j] = subZ
                    sio.savemat(DataRoad + 'X/' + files[i], {'X': X})
                    sio.savemat(DataRoad + 'Y/' + files[i], {'Y': Y})
                    sio.savemat(DataRoad + 'Z/' + files[i], {'Z': Z})
                break

        else:
            print('Using the prepared testset and initial values in folder data/Harvard/test')

    def all_test_data_in(self, idx):
        for root, dirs, files in os.walk('data/UH/test/Y/'):
            filename = files[idx]
            data = sio.loadmat('data/UH/test/Y/' + filename)
            Y = data['Y']
            data = sio.loadmat('data/UH/test/Z/' + filename)
            Z = data['Z']

        return Y, Z, filename