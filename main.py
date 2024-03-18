#!/usr/bin/env python
# coding: utf-8

# In[1]:


import utility
import data
import model
import loss
from option import args
from checkpoint import Checkpoint
from trainer import Trainer


if __name__=='__main__':
    utility.set_seed(args.seed)
    checkpoint = Checkpoint(args)

    if checkpoint.ok:
        # 载入了数据
        loader = data.Data(args)
        # 载入了模型
        model = model.Model(args, checkpoint)
        # 损失函数
        loss = loss.Loss(args)
        # 创捷一个Trainer类 t
        t = Trainer(args, loader, model, loss, checkpoint)
        # train 和 test 的过程
        while not t.terminate():
            t.train()
            t.test()
        # 关闭logfile
        checkpoint.done()







