from accelerate import Accelerator
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from tqdm import tqdm
from utils import set_seed, time_logger, get_batch, load_data, load_data_with_genre, get_genre_batch
from models_.main_model import MOESD, MOESD_trm, soft_MOESD, base, genre_smoe, strm, m_smoe
from models_.graph_encoder.models import SimpleHGN
from config import cfg, update_cfg
from datetime import datetime
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class Trainer:
    def __init__(self,
                 cfg,
                 optimizer=torch.optim.AdamW,
                 lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
                 ):

        self.num_hops = cfg.train.num_hops

        self.device = torch.device(cfg.device)

        self.batch_size = cfg.train.batch_size
        self.data, self.genre_map = load_data_with_genre(cfg)
        self.genre = nn.Parameter(torch.randn((29, 1030)))
        # self.data = load_data(cfg)
        self.train_mask = self.data.train_mask
        self.val_mask = self.data.val_mask
        self.test_mask = self.data.test_mask

        self.review_movie_map = self.data.review_movie_map
        self.review_user_map = self.data.review_user_map

        self.epochs = cfg.train.epochs
        self.lr = cfg.train.lr
        self.weight_decay = cfg.train.weight_decay

        self.model_name = cfg.model.name
        if self.model_name == 'MOESD_trm':
            self.model = MOESD_trm(cfg=cfg).to(self.device)
        elif self.model_name == 'MOESD':
            self.model = MOESD(cfg=cfg).to(self.device)
        elif self.model_name == 'smoe':
            self.model = soft_MOESD(cfg=cfg).to(self.device)
        elif self.model_name == 'base':
            self.model = base(cfg=cfg).to(self.device)
        elif self.model_name == 'genre_smoe':
            self.model = genre_smoe(cfg=cfg).to(self.device)
        elif self.model_name == 'strm':
            self.model = strm(cfg=cfg).to(self.device)
        elif self.model_name == 'msmoe':
            self.model = m_smoe(cfg=cfg).to(self.device)

        self.model = genre_smoe(cfg=cfg).to(self.device)
        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)
        print(f"\nNumber of parameters: {trainable_params}")
        
        weight = torch.FloatTensor([1, cfg.train.weight]).to(self.device)
        self.loss_func = torch.nn.CrossEntropyLoss(weight=weight)

        # self.opt = optimizer(self.model.parameters(),
        #                      lr=self.lr,
        #                      weight_decay=self.weight_decay)
        self.opt = optimizer([{'params': self.model.parameters()},
                              {'params': self.genre}],
                             lr=self.lr,
                             weight_decay=self.weight_decay)
        # if lr_scheduler:
        #     self.lr_scheduler = lr_scheduler(optimizer=self.opt,
        #                                      T_max=5)
        # else:
        #     self.lr_scheduler = None

        self.best_f1 = 0
        self.best_auc = 0
        self.best_acc = 0
        
        self.save_dir = cfg.train.save_dir

    def save_model(self):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict()
            # "lr_scheduler": self.lr_scheduler.state_dict(),
            # "global_time_step": self.global_time_step,
        }, f'{self.save_dir}.pth')
        

    @ time_logger
    def train(self):
        self.model.train()

        mask = self.train_mask
        num_batch = torch.div(len(mask), self.batch_size,
                              rounding_mode='trunc')+1
        for epoch in range(0, self.epochs):
            y_pred = []
            y_true = []
            losses = []

            for i in tqdm(range(num_batch), desc=f"Epoch:{epoch+1}"):
                if (i+1)*self.batch_size <= len(mask):
                    # time = datetime.now()
                    # print(time)
                    batch = get_genre_batch(
                        data=self.data,
                        genre=self.genre,
                        num_hops=self.num_hops,
                        sub_nodes=mask[i *
                                       self.batch_size: (i+1)*self.batch_size],
                        review_movie_map=self.review_movie_map,
                        review_user_map=self.review_user_map,
                        movie_genre_map=self.genre_map)
                    # batch = get_batch(
                    #     data=self.data,
                    #     num_hops=self.num_hops,
                    #     sub_nodes=mask[i *
                    #                    self.batch_size: (i+1)*self.batch_size],
                    #     review_movie_map=self.review_movie_map,
                    #     review_user_map=self.review_user_map)
                else:
                    batch = get_genre_batch(
                        data=self.data,
                        genre=self.genre,
                        num_hops=self.num_hops,
                        sub_nodes=mask[i*self.batch_size:],
                        review_movie_map=self.review_movie_map,
                        review_user_map=self.review_user_map,
                        movie_genre_map=self.genre_map)
                    # batch = get_batch(
                    #     data=self.data,
                    #     num_hops=self.num_hops,
                    #     sub_nodes=mask[i *
                    #                    self.batch_size: (i+1)*self.batch_size],
                    #     review_movie_map=self.review_movie_map,
                    #     review_user_map=self.review_user_map)

                # time = datetime.now()
                # print(time)
                print(self.genre[:1][:10])
                if self.model_name == 'MOESD':
                    # out, BL_loss = self.model(batch.to(self.device))
                    out, BL_loss = self.model(batch.to(self.device))
                    CE_loss = self.loss_func(out, batch.y)
                    loss = CE_loss + BL_loss
                else:
                    out = self.model((batch[0].to(self.device), batch[1]))
                    # time = datetime.now()
                    # print(time)
                    # out = self.model(batch.to(self.device))
                    CE_loss = self.loss_func(out, batch[0].y)
                    loss = CE_loss

                loss.backward()
                print(self.genre.grad)
                # add gradient clip
                nn.utils.clip_grad_value_(self.model.parameters(), 1000)
                self.opt.step()
                self.opt.zero_grad()
                # self.lr_scheduler.step()

                y_pred.append(out.argmax(dim=1))
                y_true.append(batch[0].y)
                losses.append(loss.item())

                if i % (torch.div(num_batch, 4,
                                  rounding_mode='trunc')) == 0 and i != 0 and epoch >= 2:
                    self.valid_test('test')
            # self.lr_scheduler.step()
            y_pred = [tensor.cpu().numpy() for tensor in y_pred]
            y_pred = np.concatenate(y_pred, axis=0)

            y_true = [tensor.cpu().numpy() for tensor in y_true]
            y_true = np.concatenate(y_true, axis=0)

            test_loss = sum(losses) / num_batch

            f1 = f1_score(y_true, y_pred)
            auc = roc_auc_score(y_true=y_true, y_score=y_pred)
            acc = accuracy_score(y_true=y_true, y_pred=y_pred)

            print(
                f"Epoch {epoch+1}: train loss: {test_loss:.4f}, f1: {f1:.4f}, auc: {auc:.4f}, acc: {acc:.4f}")

            self.valid_test('test')

        self.save_model()

    @torch.no_grad()
    def valid_test(self, step):
        self.model.eval()

        y_pred = []
        y_true = []
        losses = []

        if step == 'valid':
            # loader = self.val_loader
            mask = self.val_mask
        elif step == 'test':
            # loader = self.test_loader
            mask = self.test_mask

        num_batch = torch.div(len(mask), self.batch_size,
                              rounding_mode='trunc')+1

        for i in tqdm(range(num_batch)):
            if (i+1)*self.batch_size <= len(mask):
                # batch = get_batch(
                #     data=self.data,
                #     num_hops=self.num_hops,
                #     sub_nodes=mask[i * self.batch_size: (i+1)*self.batch_size],
                #     review_movie_map=self.review_movie_map,
                #     review_user_map=self.review_user_map)
                batch = get_genre_batch(
                    data=self.data,
                    genre=self.genre,
                    num_hops=self.num_hops,
                    sub_nodes=mask[i*self.batch_size:(i+1)*self.batch_size],
                    review_movie_map=self.review_movie_map,
                    review_user_map=self.review_user_map,
                    movie_genre_map=self.genre_map)
            else:
                # batch = get_batch(
                #     data=self.data,
                #     num_hops=self.num_hops,
                #     sub_nodes=mask[i*self.batch_size:],
                #     review_movie_map=self.review_movie_map,
                #     review_user_map=self.review_user_map)
                batch = get_genre_batch(
                    data=self.data,
                    genre=self.genre,
                    num_hops=self.num_hops,
                    sub_nodes=mask[i*self.batch_size:],
                    review_movie_map=self.review_movie_map,
                    review_user_map=self.review_user_map,
                    movie_genre_map=self.genre_map)

            if self.model_name == 'MOESD':
                out, BL_loss = self.model(batch.to(self.device))
                CE_loss = self.loss_func(out, batch.y)
                loss = CE_loss + BL_loss
            else:
                out = self.model((batch[0].to(self.device), batch[1]))
                # out = self.model(batch.to(self.device))
                CE_loss = self.loss_func(out, batch[0].y)
                loss = CE_loss

            y_pred.append(out.argmax(dim=1))
            y_true.append(batch[0].y)
            losses.append(loss.item())

        y_pred = [tensor.cpu().numpy() for tensor in y_pred]
        y_pred = np.concatenate(y_pred, axis=0)

        y_true = [tensor.cpu().numpy() for tensor in y_true]
        y_true = np.concatenate(y_true, axis=0)

        test_loss = sum(losses) / (torch.div(len(mask),
                                             self.batch_size, rounding_mode='trunc')+1)
        f1 = f1_score(y_true, y_pred)

        acc = accuracy_score(y_true=y_true, y_pred=y_pred)

        auc = roc_auc_score(y_true=y_true, y_score=y_pred)

        if step == 'valid':
            print(
                f"         {step} loss: {test_loss:.4f}, f1: {f1:.4f}, auc: {auc:.4f}, acc: {acc:.4f}")
        else:
            if f1 > self.best_f1:
                self.best_f1 = f1
            if acc > self.best_acc:
                self.best_acc = acc
            if auc > self.best_auc:
                self.best_auc = auc
            print(
                f"         {step}  loss: {test_loss:.4f}, f1: {f1:.4f}, auc: {auc:.4f}, acc: {acc:.4f}, best_f1: {self.best_f1:.4f}. best_auc: {self.best_auc:.4f}, best_acc: {self.best_acc:.4f}")


def main():
    set_seed(cfg.seed)
    trainer = Trainer(cfg)
    trainer.train()
    # trainer.valid_test('test')


if __name__ == "__main__":
    cfg = update_cfg(cfg)
    print(
        f"---------------------------Starting training seed={cfg.seed}-----------------------------")
    print(cfg)
    main()
