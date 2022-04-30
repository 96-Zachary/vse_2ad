def adjust_learning_rate(args, optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * (0.1 ** (epoch // args.lr_update))