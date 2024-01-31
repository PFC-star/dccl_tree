def dua(model, data_loader):
    """ Test step """
    model.eval()
    mom_pre = 0.1
    decay_factor = 0.94
    acc_best = 0
    mom_best = 0
    # model_best = model.copy()
    for batch_idx, (data, target) in enumerate(data_loader):
        if batch_idx == 100:
            break
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        mom_new = (mom_pre * decay_factor)
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.train()
                m.momentum = mom_new + 0.005
                # m.momentum = 0.01342
        mom_pre = mom_new
        _ = model(data)
        acc = test(model, data_loader)
        if acc >= acc_best:
            # model_best = model.copy()
            acc_best = acc
            mom_best = mom_pre + 0.005
            print('acc_best:'+str(acc_best)+' ----  momentum_best:'+str(mom_best))
    return model