import os
import scipy.io as sio
import torch
import torch.nn as nn
import model.SSFT

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

data = sio.loadmat('data/CAVE/response coefficient')
R = data['R']


class Model(nn.Module):
    def __init__(self, opt, ckp):
        super(Model, self).__init__()
        ckp.write_log(f"Making model CatFFT")
        # self.device = torch.device('cpu' if opt.cpu else 'cuda')

        self.device = torch.device("cuda:%s" % opt.gpu_ids[0]
                                   if torch.cuda.is_available() and len(opt.gpu_ids) > 0
                                   else "cpu")


        # self.model = torch.nn.DataParallel(SSFT.make_model(opt), device_ids=[0,1] ).cuda() #to(self.device)

        self.model = CatNet.make_model(opt).to(self.device)



        if opt.test_only:
            self.load(opt.pre_train, cpu=opt.cpu)

        # compute parameter
        num_parameter = self.count_parameters(self.model)
        ckp.write_log(f"The number of parameters is {num_parameter / 1000 ** 2:.2f}M")

    def forward(self, x):
         return self.model(x)

    def get_model(self):
        return self.model

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def save(self, path, is_best=False):
        target = self.get_model()
        torch.save(
            target.state_dict(),
            os.path.join(path, 'model', 'model_latest.pt')
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(path, 'model', 'model_best.pt')
            )

    def load(self, pre_train='.', cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}
        #### load primal model ####
        if pre_train != '.':
            print('Loading model from {}'.format(pre_train))
            self.get_model().load_state_dict(
                torch.load(pre_train, **kwargs),
                strict=False
            )