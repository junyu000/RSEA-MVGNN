import torch
import numpy as np


class early_stopping:
    def __init__(self, patience, verbose=False, delta=0, save_path='checkpoint.ckpt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0

        self.best_score = None
        self.best_score1 = None

        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, first_loss, model):
        score = -val_loss
        first_loss = -first_loss
        if self.best_score is None:
            self.best_score = score
            self.best_score1 = first_loss
            self.save_checkpoint(val_loss, model)
        # elif score < self.best_score - self.delta and first_loss < self.best_score1:
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score             
            self.best_score1 = first_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min} --> {val_loss}).  Saving model ...')
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss