from torch.utils.data import DataLoader
from importlib import import_module


class Data:
    def __init__(self, args):

        m = import_module('data.' + args.dataset.lower())

        if args.dataset == 'CAVE':
            trainset = getattr(m, 'TrainSet')(args)
            self.loader_train = DataLoader(
                trainset,
                batch_size=args.batch_size,
                num_workers=args.n_threads,
                shuffle=True,
                pin_memory=not args.cpu
            )
            testset = getattr(m, 'TestSet')(args)
            self.loader_test = DataLoader(
                testset,
                batch_size=1,
                num_workers=1,
                shuffle=False,
                pin_memory=not args.cpu
            )

        elif args.dataset in ['Harvard', 'UH']:
            testset = getattr(m, 'TestSet')(args)
            self.loader_test = DataLoader(
                testset,
                batch_size=1,
                num_workers=1,
                shuffle=False,
                pin_memory=not args.cpu
            )

        else:
            raise SystemExit('Error: no such type of dataset!')