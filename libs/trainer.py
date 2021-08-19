from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from train_utils import AverageMeter, get_num_parameters


class Trainer:
    def __init__(self, model, criterion, optimizer, config, device, train_data_loader, valid_data_loader=None, lr_scheduler=None, early_stopper=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.early_stopper = early_stopper

        self.train_loss = AverageMeter("Train Loss")
        self.valid_loss = AverageMeter("Validation Loss")
        self.writer = SummaryWriter(log_dir=self.config["TB_LOGDIR"])

        self.stop_training = False

    def on_train_begin(self):
        print(f"Number of Total Parameters: {get_num_parameters(self.model)}")
        summary(self.model)
        # Model Parallel
        return

    def on_train_end(self):
        # Save final result
        return

    def on_valid_begin(self):
        return

    def on_valid_end(self):
        return

    def on_train_batch_begin(self, batch):
        # Maybe visualize?
        return

    def on_train_batch_end(self, batch):
        return

    def on_epoch_begin(self, epoch):
        return

    def on_epoch_end(self, epoch):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if self.early_stopper(self.train_loss.avg, self.valid_loss.avg, self.model):
            self.stop_training = True

        self.writer.add_scalars("Loss", {"Train Loss": self.train_loss.avg, "Validation Loss": self.valid_loss.avg}, global_step=epoch)
        print(f"Epoch - {epoch} Train Loss - {self.train_loss.avg:.10f} Valid Loss - {self.valid_loss.avg:.10f}")

        self.train_loss.reset()
        self.valid_loss.reset()

        return

    def on_valid_batch_begin(self, batch):
        return

    def on_valid_batch_end(self, batch):
        return

    def train_one_epoch(self, epoch):

        for i, (data, target) in enumerate(self.train_data_loader):
            self.on_train_batch_begin((data, target))

            data, target = data.to(self.device), target.to(self.device)
            out = self.model(data)
            loss = self.criterion(out, target)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.train_loss.update(float(loss))

            print(f"Epoch - {epoch} Iter - {i} Train Loss - {float(self.train_loss.val):.10f}")

            self.on_train_batch_end((data, target))


    def train(self, num_epochs):
        self.on_train_begin()

        for epoch in range(num_epochs):
            self.on_epoch_begin(epoch)

            self.train_one_epoch(epoch)
            self.model.eval()
            self.validate()
            self.model.train()

            self.on_epoch_end(epoch)

        self.on_train_end()

    def validate(self)
        try:
            assert self.valid_data_loader is not None, "ERROR: Validation data loader is not given"
        except AssertionError:
            return

        self.on_valid_begin()

        for i, (data, target) in enumerate(self.valid_data_loader):
            self.on_valid_batch_begin((data, target))

            data, target = data.to(self.device), target.to(self.device)
            out = self.model(data)
            loss = self.criterion(out, target)
            self.valid_loss.update(float(loss))

            self.on_valid_batch_end((data, target))

        self.on_valid_end()
