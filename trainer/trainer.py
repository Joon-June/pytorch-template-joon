from tqdm import tqdm

from config import Config
from utils import (
    AverageMeter,
    get_num_parameters,
    set_seeds,
    ModelEval,
    Writer,
    batch_visualize,
    save_checkpoint,
)


class Trainer:
    def __init__(
        self,
        config: Config,
        model,
        criterion,
        optimizer,
        device,
        train_dataloader,
        eval_dataloader=None,
        writer: Writer = None,
    ):
        self.config = config

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_dataloader = train_dataloader

        self.eval_dataloader = eval_dataloader
        self.writer = writer

        self.train_loss = AverageMeter("Train Loss")
        self.eval_loss = AverageMeter("Eval Loss")

        self.log = self.writer is not None
        self.eval = self.config.eval_every > 0
        self.viz = self.config.visualize_every > 0
        self.save = self.config.save_every > 0

        self.global_step = 0
        self.step = 0

    def on_train_begin(self):
        # Model Parallel
        self.model.to(self.device)

        # Other initializations
        set_seeds(self.config.seed)

        # Initial logging
        print(f"Number of Total Parameters: {get_num_parameters(self.model)}")

    def on_train_end(self):
        # Save all final results
        return

    def on_train_batch_begin(self, it, batch):
        # Visualize
        if self.viz and (it + 1) % self.config.visualize_every == 0:
            batch_visualize(batch)

    def on_train_batch_end(self, it, batch):
        # Update steps
        self.global_step += len(batch[0])
        self.step += 1

        if self.log:
            self.writer.log({"train/loss": self.train_loss.val}, step=self.global_step)

        if self.eval and it % self.config.eval_every == 0:
            with ModelEval(self.model) as m:
                self.evaluate(m)

        if self.save and it % self.config.save_every == 0:
            save_checkpoint(self.model)

    def on_eval_begin(self):
        return

    def on_eval_end(self):
        if self.log:
            self.writer.log({"eval/loss": self.eval_loss.avg}, step=self.global_step)

        self.eval_loss.reset()

    def on_eval_batch_begin(self):
        return

    def on_eval_batch_end(self):
        return

    def train(self):
        self.on_train_begin()

        pbar = tqdm(range(self.config.num_iters))

        for it in pbar:
            x, y = next(self.train_dataloader)
            self.on_train_batch_begin(it, (x, y))

            out = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.train_loss.update(float(loss))
            pbar.set_description(
                f"Train: Iter - {it} Loss - {float(self.train_loss.val):.8f}"
            )

            self.on_train_batch_end(it, (x, y))

        self.on_train_end()

    def evaluate(self, model):
        if self.eval_dataloader is None:
            return

        self.on_eval_begin()

        pbar = enumerate(tqdm(self.eval_dataloader))

        for i, (x, y) in pbar:
            self.on_eval_batch_begin()

            x, y = x.to(self.device), y.to(self.device)
            out = model(x)
            loss = self.criterion(out, y)
            self.eval_loss.update(float(loss))

            self.on_eval_batch_end()

        self.on_eval_end()
