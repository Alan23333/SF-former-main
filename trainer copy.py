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
                sr = utility.Upsample(utility.hyperConvert2D(sr), Y, Z, self.opt.scale, self.opt.dataset, pad)  # sr(512,512,29)

                psnr = utility.PSNR(X, sr, self.opt.data_range)  # (512,512,29)
                eval_psnr += psnr

                ssim = utility.SSIM(X, sr, self.opt.data_range)
                eval_ssim += ssim

                sam = utility.SAM(X, sr)
                eval_sam += sam

                ergas = utility.ERGAS(X, sr, self.opt.scale)
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