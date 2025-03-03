from datetime import datetime
from transformers import get_cosine_schedule_with_warmup
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from tqdm import tqdm
import json
from utils import set_seed, time_logger, get_batch, load_data, get_genre_batch, get_model, MetricsHandler
from config import cfg, update_cfg
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class Data(Dataset):
    def __init__(self, name, dataset):
        super().__init__()
        if name == 'train':
            self.mask = torch.load(
                f'/data3/whr/zhk/Spoiler_Detection/Data/processed_{dataset}_data/mask/train_index.pt').to(torch.long)
            # self.mask = (torch.load(
            #     "/data3/whr/zhk/Spoiler_Detection/Data/processed_data/small_mask/train_mask.pt")).to(torch.long)
        elif name == 'valid':
            self.mask = torch.load(
                f'/data3/whr/zhk/Spoiler_Detection/Data/processed_{dataset}_data/mask/val_index.pt').to(torch.long)
            # self.mask = (torch.load(
            #     "/data3/whr/zhk/Spoiler_Detection/Data/processed_data/small_mask/valid_mask.pt")).to(torch.long)
        else:
            self.mask = torch.load(
                f'/data3/whr/zhk/Spoiler_Detection/Data/processed_{dataset}_data/mask/test_index.pt').to(torch.long)
            # self.mask = (torch.load(
            #     "/data3/whr/zhk/Spoiler_Detection/Data/processed_data/small_mask/test_mask.pt")).to(torch.long)

    def __getitem__(self, index):
        return self.mask[index]

    def __len__(self):
        return self.mask.size(0)


class Trainer:
    def __init__(self,
                 cfg,
                 optimizer=torch.optim.AdamW,
                 ):

        self.num_hops = cfg.train.num_hops

        self.device = torch.device(cfg.device)

        self.flooding = cfg.flooding
        self.multi_test = cfg.train.multi_test
        self.flooding_gate = cfg.flooding_gate
        self.batch_size = cfg.train.batch_size
        self.data = load_data(cfg).to('cpu')

        self.epochs = cfg.train.epochs
        self.lr = cfg.train.lr
        self.weight_decay = cfg.train.weight_decay
        self.model_name = cfg.model.name

        self.model = get_model(cfg.model.name, cfg).to(self.device)

        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)
        print(
            f"---------------------------Starting training seed={cfg.seed}-----------------------------")
        print(cfg)
        print(f"\nNumber of parameters: {trainable_params}")

        self.use_moe = cfg.model.use_two_moe or cfg.model.use_one_moe

        weight = torch.FloatTensor([1, cfg.train.weight]).to(self.device)
        self.loss_func = torch.nn.CrossEntropyLoss(weight=weight)

        self.opt = optimizer(self.model.parameters(),
                             lr=self.lr,
                             weight_decay=self.weight_decay,
                             amsgrad=True)

        self.metrics_handler = MetricsHandler(metrics=[('f1', True),
                                                       ('auc', True),
                                                       ('acc', True)],
                                              patience=cfg.train.patience,
                                              checkpoint_dir=cfg.train.save_dir,
                                              train_name=cfg.train.name,
                                              seed=cfg.seed,
                                              warmup_epochs=cfg.train.warmup_epochs)

        def collate_fn(data):
            batch = get_genre_batch(
                data=self.data,
                num_hops=self.num_hops,
                sub_nodes=data)
            return batch

        train_set = Data('train', cfg.dataset)
        val_set = Data('valid', cfg.dataset)
        test_set = Data('test', cfg.dataset)
        self.train_loader = DataLoader(
            train_set, collate_fn=collate_fn, shuffle=True, batch_size=self.batch_size)
        self.val_loader = DataLoader(
            val_set, collate_fn=collate_fn, batch_size=self.batch_size)
        self.test_loader = DataLoader(
            test_set, collate_fn=collate_fn, batch_size=self.batch_size)

        self.test_per_epoch = cfg.train.test_per_epoch

    @ time_logger
    def train(self):
        self.model.train()

        loader = self.train_loader
        for epoch in range(0, self.epochs):
            y_pred = []
            y_true = []
            losses = []

            for batch in tqdm(loader, desc=f'Epoch:{epoch+1}'):
                out = self.model(batch.to(self.device))
                CE_loss = self.loss_func(out, batch.y)
                loss = CE_loss

                # add gradient clip
                nn.utils.clip_grad_value_(self.model.parameters(), 1000)

                self.opt.step()
                self.opt.zero_grad()

                y_pred.append(out.argmax(dim=1))
                y_true.append(batch.y)
                losses.append(loss.item())

            y_pred = [tensor.cpu().numpy() for tensor in y_pred]
            y_pred = np.concatenate(y_pred, axis=0)

            y_true = [tensor.cpu().numpy() for tensor in y_true]
            y_true = np.concatenate(y_true, axis=0)

            test_loss = sum(losses) / len(loader)
            f1, auc, acc = self.compute_metrics(y_pred, y_true)

            print(
                f"Epoch {epoch+1}: train loss: {test_loss:.4f}, f1: {f1:.4f}, auc: {auc:.4f}, acc: {acc:.4f}")

            # self.valid_test('valid')
            early_stop = self.valid_test('test', epoch)
            if early_stop == True:
                break

    @torch.no_grad()
    def valid_test(self, step, epoch):
        self.model.eval()

        y_pred = []
        y_true = []
        losses = []

        if step == 'valid':
            loader = self.val_loader
        elif step == 'test':
            loader = self.test_loader

        for batch in tqdm(loader):
            out = self.model(batch.to(self.device))

            CE_loss = self.loss_func(out, batch.y)
            loss = CE_loss

            y_pred.append(out.argmax(dim=1))
            y_true.append(batch.y)
            losses.append(loss.item())

        y_pred = [tensor.cpu().numpy() for tensor in y_pred]
        y_pred = np.concatenate(y_pred, axis=0)

        y_true = [tensor.cpu().numpy() for tensor in y_true]
        y_true = np.concatenate(y_true, axis=0)

        test_loss = sum(losses) / len(loader)
        f1, auc, acc = self.compute_metrics(y_pred, y_true)

        early_stop = self.metrics_handler.handle_metrics(metrics=[('f1', f1),
                                                                  ('auc', auc),
                                                                  ('acc', acc)],
                                                         model=self.model,
                                                         epoch=epoch,
                                                         test_loss=test_loss,
                                                         step=step)
        return early_stop

    def compute_metrics(self, y_pred, y_true):
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true=y_true, y_score=y_pred)
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        return f1, auc, acc


def main():
    set_seed(cfg.seed)
    trainer = Trainer(cfg)
    trainer.train()
    # trainer.valid_test('test')


if __name__ == "__main__":
    cfg = update_cfg(cfg)
    main()
