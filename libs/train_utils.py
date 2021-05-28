class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class EarlyStopping:
    def __init__(self, patience=5, eps=1e-3):
        self.patience = patience
        self.eps = eps
        self.best_train_loss = np.Inf
        self.best_val_loss = np.Inf
        self.count = 0
        self.early_stop = False

    def __call__(self, train_loss, val_loss, model):
        if val_loss < self.best_val_loss - self.eps:
            self.best_val_loss = val_loss
            self.count = 0
            self.save_checkpoint(model)
        else: # Model is not improving
            self.count += 1
            print(f"Early stopping count {self.count}/{self.patience}")
            if self.count >= self.patience and self.best_train_loss - train_loss < self.eps:
                print(f"Early stopping activated.")
                self.early_stop = True

        if train_loss < self.best_train_loss - self.eps:
            self.best_train_loss = train_loss

        return self.early_stop

    @staticmethod
    def save_checkpoint(model):
        raise NotImplementedError
        print("Saved a model at " + model_name)


def get_num_parameters(model):
    return sum(p.numel() for p in model.parameters())