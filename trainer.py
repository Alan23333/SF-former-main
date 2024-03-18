import numpy as np
import torch
import utility
from decimal import Decimal
from tqdm import tqdm
import datetime
import cv2
from thop import profile
import time


class Trainer():
    def __init__(self, opt, loader, my_model, my_loss, ckp):
        self.opt = opt
        self.scale = opt.scale
        self.ckp = ckp
        if not self.opt.test_only:
            self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        # self.device = torch.device('cpu' if self.opt.cpu else 'cuda')
        self.device = torch.device("cuda:%s" % opt.gpu_ids[0] if torch.cuda.is_available() and len(opt.gpu_ids) > 0 else "cpu")
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(opt, self.model)
        self.scheduler = utility.make_scheduler(opt, self.optimizer)
        self.error_last = 1e8

    def train(self):
        # 设置epoch从从哪里开始，last_epoch + 1 轮的训练，是用来断点继续训练的
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_last_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}\t{}'.format(epoch, Decimal(lr), datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        )
        self.loss.start_log()
        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (C, CGT) in enumerate(self.loader_train):
            C = C.to(self.device)
            CGT = CGT.to(self.device)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()

            # forward
            C = self.model(C)

            # compute loss
            loss = self.loss(C, CGT)

            if loss.item() < self.opt.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.opt.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s\t{}'.format(
                    (batch + 1) * self.opt.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release(),
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.step()

    def test(self):
        epoch = self.scheduler.last_epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, 1))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            eval_psnr = 0
            eval_ssim = 0
            eval_sam = 0
            eval_ergas = 0
            tqdm_test = tqdm(self.loader_test, ncols=80)
            timer_model = utility.timer()
            for _, (
                    X,
                    Y, Z, D, A,
                    filename, pad
            ) in enumerate(tqdm_test):# A(1,10,512,512) D(1,29,10) X(1,512,512,29) Y(1,512,512,3) Z(1,64,64,29)
                # print(filename)
                filename = filename[0]
                A = A.to(self.device)
                # Xes = Xes.to(self.device)
                timer_model.tic()

                # A = A[0].permute(1, 2, 0).cpu().numpy()
                D = D[0].numpy() # D(29,10)

                # sr = np.dot(D, utility.hyperConvert2D(A))
                # sr = utility.reshape(sr.T, (512, 512, 29))
                # sr = utility.reshape(sr.T, (1040, 1040, 29))

                # flops, params = profile(self.model, (A,))
                # print('the flops is {}G, the params is {}M'.format(round(flops / (10 ** 9), 2), round(params / (10 ** 6), 2)))

                # start_time = time.time()
                # forward
                C = self.model(A) # C(1,10,512,512)
                # end_time = time.time()
                # elapsed_time = end_time - start_time
                # print(f"运行时间: {elapsed_time} 秒")

                # sr = self.model(Xes)

                X = X[0].numpy()  # X(512,512,29)
                Y = Y[0].numpy()  # Y(512,512,3)
                Z = Z[0].numpy()  # Z(64,64,29)

                # sr = cv2.resize(Z, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)

                C = C[0].permute(1, 2, 0).cpu().numpy() #C(512,512,10)
                # sr = sr[0].permute(1, 2, 0).cpu().numpy()

                # D(29,10) C(512,512,10) -> sr(29,262144)
                sr = np.dot(D, utility.hyperConvert2D(C))
                sr = utility.reshape(sr.T, (pad, pad, 29)) # sr(512,512,29)
                # sr = utility.reshape(sr.T, (640, 640, 29))

                # sr0 = utility.reshape(sr.T, (256, 256, 93))
                sr = utility.Upsample(utility.hyperConvert2D(sr), Y, Z, self.opt.scale, self.opt.dataset, pad)  # sr(512,512,29) numpy


                psnr = utility.PSNR(X, sr, self.opt.data_range)
                eval_psnr += psnr

                ssim = utility.SSIM(X, sr, self.opt.data_range)
                eval_ssim += ssim

                sam = utility.SAM(X, sr)
                eval_sam += sam

                ergas = utility.ERGAS(sr, X, self.opt.scale)
                eval_ergas += ergas


                # save test results
                if self.opt.save_results:
                    # filename = 'PU0.mat'
                    self.ckp.save_results_nopostfix(filename, sr, self.scale)

                # print(f'{filename}:  psnr={psnr:.2f}  ssim={ssim:.4f}  sam={sam:.2f}  ergas={ergas:.2f}')

            self.ckp.log[-1, 0] = eval_psnr / len(self.loader_test)
            eval_ssim = eval_ssim / len(self.loader_test)
            eval_sam = eval_sam / len(self.loader_test)
            eval_ergas = eval_ergas / len(self.loader_test)
            best = self.ckp.log.max(0)
            # self.ckp.write_log(
            #     '[{} x{}]\tPSNR: {:.2f} (Best: {:.2f} @epoch {})'.format(
            #         self.opt.dataset, self.scale,
            #         self.ckp.log[-1, 0],
            #         best[0][0],
            #         best[1][0] + 1
            #     )
            # )
            self.ckp.write_log(
                '[{} x{}]\tPSNR: {:.2f}\tSSIM: {:.4f}\tSAM: {:.2f}\tERGAS: {:.2f} (Best: {:.2f} @epoch {})'.format(
                    self.opt.dataset, self.scale,
                    self.ckp.log[-1, 0],
                    eval_ssim,
                    eval_sam,
                    eval_ergas,
                    best[0][0],
                    best[1][0] + 1
                )
            )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        if not self.opt.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def step(self):
        self.scheduler.step()

    def terminate(self):
        if self.opt.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch
            return epoch >= self.opt.epochs


def _as_floats(im1, im2):
    float_type = np.result_type(im1.dtype, im2.dtype, np.float32)
    im1 = np.asarray(im1, dtype=float_type)
    im2 = np.asarray(im2, dtype=float_type)
    return im1, im2


def compare_mse(im1, im2):
    im1, im2 = _as_floats(im1, im2)
    return np.mean(np.square(im1 - im2), dtype=np.float64)


def compare_psnr(im_true, im_test, data_range=None):
    im_true, im_test = _as_floats(im_true, im_test)

    err = compare_mse(im_true, im_test)
    return 10 * np.log10((data_range ** 2) / err)


def PSNR(reference, target, data_range):
    bands = reference.shape[2]
    mpsnr = 0
    for i in range(bands):
        mpsnr += compare_psnr(reference[:, :, i], target[:, :, i], data_range=data_range)
    mpsnr /= bands
    return mpsnr


def SAM_CPU(im_true, im_fake):
    I_true = im_true.data.cpu().numpy()
    I_fake = im_fake.data.cpu().numpy()
    N = I_true.shape[0]
    C = I_true.shape[1]
    H = I_true.shape[2]
    W = I_true.shape[3]
    batch_sam = 0
    for i in range(N):
        true = I_true[i,:,:,:].reshape(C, H*W)
        fake = I_fake[i,:,:,:].reshape(C, H*W)
        nom = np.sum(np.multiply(true, fake), 0).reshape(H*W, 1)
        denom1 = np.sqrt(np.sum(np.square(true), 0)).reshape(H*W, 1)
        denom2 = np.sqrt(np.sum(np.square(fake), 0)).reshape(H*W, 1)
        sam = np.arccos(np.divide(nom,np.multiply(denom1,denom2))).reshape(H*W, 1)
        sam = sam/np.pi*180
        # ignore pixels that have zero norm
        idx = (np.isfinite(sam))
        batch_sam += np.sum(sam[idx])/np.sum(idx)
        if np.sum(~idx) != 0:
            print("waring: some values were ignored when computing SAM")
    return batch_sam/N


def calc_ergas(img_tgt, img_fus):  # (1,29,512,512) tensor
    scale = 8
    img_tgt = img_tgt.squeeze(0).data.cpu().numpy()  # (29,512,512) numpy
    img_fus = img_fus.squeeze(0).data.cpu().numpy()  # (29,512,512) numpy
    img_tgt = np.squeeze(img_tgt)  # (29,512,512) numpy
    img_fus = np.squeeze(img_fus)  # (29,512,512) numpy
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)  # (29,262144) numpy
    img_fus = img_fus.reshape(img_fus.shape[0], -1)  # (29,262144) numpy

    rmse = np.mean((img_tgt-img_fus)**2, axis=1)  # (29) numpy
    rmse = rmse**0.5  # (29) numpy
    mean = np.mean(img_tgt, axis=1)  # (29) numpy

    ergas = np.mean((rmse/mean)**2)  # (1)
    ergas = 100/scale*ergas**0.5  # (1) numpy

    return ergas


def ERGAS(references, target, ratio):  # (512,512,29)
    rows, cols, bands = references.shape  # (512,512,29)
    d = 1 / ratio  # 1/8
    pixels = rows * cols  # 262144
    ref_temp = np.reshape(references, [pixels, bands], order='F')  # (262144,29)
    tar_temp = np.reshape(target, [pixels, bands], order='F')  # (262144,29)
    err = ref_temp - tar_temp  # (262144,29)
    rmse2 = np.sum(err ** 2, axis=0) / pixels  # (29)
    uk = np.mean(tar_temp, axis=0)  # (29)
    relative_rmse2 = rmse2 / uk ** 2  # (29)
    total_relative_rmse = np.sum(relative_rmse2)  # (1)
    out = 100 * d * np.sqrt(1 / bands * total_relative_rmse)  # (1)
    return out