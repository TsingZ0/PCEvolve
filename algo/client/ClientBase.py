import copy
import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.dataset import get_real_data
from utils.model import get_model
from models.clip import get_clip
from models.inception import get_inception
from models.vits import get_vit
from models.resnets import get_resnet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torcheval.metrics import FrechetInceptionDistance


class ClientBase(object):
    def __init__(self, args):
        self.args = args
        self.it = 0

        get_real_data(args)
        self.eval_loader = self.load_eval_dataset()
        self.current_volume_per_label = [0 for _ in range(args.num_labels)]
        self.done = False

        self.model = get_model(args)
        numel = 0
        for param in self.model.parameters():
            numel += param.numel()
        print('Number of client model parameters:', numel)
        self.opt = torch.optim.AdamW(self.model.head.parameters(), lr=self.args.client_learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, args.client_epochs)
        self.loss = torch.nn.CrossEntropyLoss()

        if 'CLIP' in args.client_use_embedding:
            args.client_model_pretrained = True
            self.encoder = get_clip(args)
        elif 'Inception' in args.client_use_embedding:
            args.client_model_pretrained = True
            self.encoder = get_inception(args)
        elif 'ViT' in args.client_use_embedding:
            args.client_model_pretrained = True
            self.encoder = get_vit(args)
        elif 'ResNet' in args.client_use_embedding:
            args.client_model_pretrained = True
            self.encoder = get_resnet(args)
        else:
            self.encoder = self.model.encoder
        self.encoder.eval()


    def send(self, content=None):
        if content == 'real':
            real_data = self.load_real_dataset(is_raw=True)
            return copy.deepcopy(self.current_volume_per_label), self.done, real_data
        else:
            return copy.deepcopy(self.current_volume_per_label), self.done

    def receive(self):
        if not self.args.use_generated:
            generated_dataset_dir = os.path.join(self.args.dataset_dir, 'generated', self.args.task)
            assert os.path.exists(generated_dataset_dir), 'generated_dataset_dir does not exist'
        train_dataset_dir = os.path.join(self.args.dataset_dir, 'train', self.args.task)
        assert os.path.exists(train_dataset_dir), 'train_dataset_dir does not exist'

    def load_eval_dataset(self, is_raw=False, batch_size=None):
        eval_dataset_dir = os.path.join(self.args.dataset_dir, 'test')
        eval_dataset = torch.load(os.path.join(eval_dataset_dir, self.args.client_dataset + '.pt'))
        if is_raw:
            return eval_dataset
        else:
            if batch_size is None:
                batch_size = self.args.client_batch_size
            return DataLoader(eval_dataset, batch_size, drop_last=False, shuffle=True)

    def load_real_dataset(self, is_raw=False, batch_size=None):
        real_dataset_dir = os.path.join(self.args.dataset_dir, f'real/{self.args.real_volume_per_label}')
        real_dataset = torch.load(os.path.join(real_dataset_dir, self.args.client_dataset + '.pt'))
        if is_raw:
            return real_dataset
        else:
            if batch_size is None:
                batch_size = self.args.client_batch_size
            return DataLoader(real_dataset, batch_size, drop_last=False, shuffle=True)

    def load_train_dataset(self, is_raw=False, is_shuffle=True, batch_size=None):
        train_dataset_dir = os.path.join(self.args.dataset_dir, 'train', self.args.task)
        try:
            train_dataset = torch.load(os.path.join(train_dataset_dir, f'{self.it}/dataset.pt'))
            if is_raw:
                return train_dataset
            else:
                if batch_size is None:
                    batch_size = self.args.client_batch_size
                return DataLoader(train_dataset, batch_size, drop_last=False, shuffle=is_shuffle)
        except FileNotFoundError:
            return None
        
    def eval_metrics(self):
        self.model.eval()

        trues_per_label = [[] for _ in range(self.args.num_labels)]
        preds_per_label = [[] for _ in range(self.args.num_labels)]
        with torch.no_grad():
            for x, y in self.eval_loader:
                x = x.to(self.args.device)
                y = y.to(self.args.device)
                output = self.model(x)
                pred = torch.argmax(output, dim=1)
                for pp, yy in zip(pred, y):
                    trues_per_label[yy.item()].append(yy.item())
                    preds_per_label[yy.item()].append(pp.item())

        trues = []
        for tr in trues_per_label:
            trues.extend(tr)
        preds = []
        for pr in preds_per_label:
            preds.extend(pr)

        train_loader = self.load_train_dataset()
        real_per_label = [[] for _ in range(self.args.num_labels)]
        gen_per_label = [[] for _ in range(self.args.num_labels)]
        if train_loader is not None:
            real_loader = self.load_real_dataset()
            FID = FrechetInceptionDistance(
                self.encoder, 
                self.encoder.feature_dim, 
                self.args.device
            )
            FIDs = [FrechetInceptionDistance(
                    self.encoder, 
                    self.encoder.feature_dim, 
                    self.args.device
                ) for _ in range(self.args.num_labels)
            ]
            with torch.no_grad():
                for x, y in real_loader:
                    FID.update(x, is_real=True)
                    for xx, yy in zip(x, y):
                        yc = yy.item()
                        FIDs[yc].update(xx.unsqueeze(0), is_real=True)
                        real_per_label[yc].append(xx.cpu().numpy())
                for x, y in train_loader:
                    FID.update(x, is_real=False)
                    for xx, yy in zip(x, y):
                        yc = yy.item()
                        FIDs[yc].update(xx.unsqueeze(0), is_real=False)
                        gen_per_label[yc].append(xx.cpu().numpy())
            FID_metrics = (FID.compute(), [f.compute() for f in FIDs])
            PSNRs = self.avg_pair_psnr(real_per_label, gen_per_label)
            PSNR_metrics = (sum(PSNRs)/len(PSNRs), PSNRs)
        else:
            FID_metrics = (0, [0 for _ in range(self.args.num_labels)])
            PSNR_metrics = (0, [0 for _ in range(self.args.num_labels)])

        return {
            'PSNR': PSNR_metrics, 
            'FID': FID_metrics, 
            'Accuracy': (accuracy_score(trues, preds), 
                [accuracy_score(t, p) 
                    for t, p in zip(trues_per_label, preds_per_label)]), 
            'Precision': (precision_score(trues, preds, average='macro', zero_division=np.nan), 
                [precision_score(t, p, average='macro', zero_division=np.nan) 
                    for t, p in zip(trues_per_label, preds_per_label)]), 
            'Recall': (recall_score(trues, preds, average='macro', zero_division=np.nan), 
                [recall_score(t, p, average='macro', zero_division=np.nan) 
                    for t, p in zip(trues_per_label, preds_per_label)]),
            'F1': (f1_score(trues, preds, average='macro', zero_division=np.nan), 
                [f1_score(t, p, average='macro', zero_division=np.nan) 
                    for t, p in zip(trues_per_label, preds_per_label)])
        }

    def train_zero_metrics(self, train_loader):
        self.model.eval()
        if train_loader is None:
            return {}

        loss = 0
        num = 0
        trues_per_label = [[] for _ in range(self.args.num_labels)]
        preds_per_label = [[] for _ in range(self.args.num_labels)]
        with torch.no_grad():
            for x, y in train_loader:
                x = x.to(self.args.device)
                y = y.to(self.args.device)
                output = self.model(x)
                loss += self.loss(output, y).item() * y.shape[0]
                num += y.shape[0]
                pred = torch.argmax(output, dim=1)
                for pp, yy in zip(pred, y):
                    trues_per_label[yy.item()].append(yy.item())
                    preds_per_label[yy.item()].append(pp.item())

        trues = []
        for tr in trues_per_label:
            trues.extend(tr)
        preds = []
        for pr in preds_per_label:
            preds.extend(pr)

        return {
            'Loss': loss/num, 
            'Accuracy': (accuracy_score(trues, preds), 
                [accuracy_score(t, p) 
                    for t, p in zip(trues_per_label, preds_per_label)]), 
            'Precision': (precision_score(trues, preds, average='macro', zero_division=np.nan), 
                [precision_score(t, p, average='macro', zero_division=np.nan) 
                    for t, p in zip(trues_per_label, preds_per_label)]), 
            'Recall': (recall_score(trues, preds, average='macro', zero_division=np.nan), 
                [recall_score(t, p, average='macro', zero_division=np.nan) 
                    for t, p in zip(trues_per_label, preds_per_label)]),
            'F1': (f1_score(trues, preds, average='macro', zero_division=np.nan), 
                [f1_score(t, p, average='macro', zero_division=np.nan) 
                    for t, p in zip(trues_per_label, preds_per_label)])
        }
    
    def train_zero(self, train_loader):
        self.model.train()
        start = time.time()

        for e in range(self.args.client_epochs):
            for x, y in train_loader:
                x = x.to(self.args.device)
                y = y.to(self.args.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            self.scheduler.step()

        print(f"\nClient training cost per iter.: {round(time.time()-start, 2)}s.")

    def train_few_metrics(self, train_loader):
        self.model.eval()
        real_loader = self.load_real_dataset()
        real_iter = iter(real_loader)
        if train_loader is None:
            return {}

        loss = 0
        num = 0
        trues_per_label = [[] for _ in range(self.args.num_labels)]
        preds_per_label = [[] for _ in range(self.args.num_labels)]
        with torch.no_grad():
            for x, y in train_loader:
                x = x.to(self.args.device)
                y = y.to(self.args.device)
                output = self.model(x)
                loss += self.loss(output, y).item() * y.shape[0]
                num += y.shape[0]
                pred = torch.argmax(output, dim=1)
                for pp, yy in zip(pred, y):
                    trues_per_label[yy.item()].append(yy.item())
                    preds_per_label[yy.item()].append(pp.item())
                # use real data
                try:
                    x_g, y_g = next(real_iter)
                except StopIteration:
                    real_iter = iter(real_loader)
                    x_g, y_g = next(real_iter)
                x_g = x_g.to(self.args.device)
                y_g = y_g.to(self.args.device)
                output_g = self.model(x_g)
                loss += self.loss(output_g, y_g).item() * y_g.shape[0]
                num += y_g.shape[0]
                pred_g = torch.argmax(output_g, dim=1)
                for pp, yy in zip(pred_g, y_g):
                    trues_per_label[yy.item()].append(yy.item())
                    preds_per_label[yy.item()].append(pp.item())

        trues = []
        for tr in trues_per_label:
            trues.extend(tr)
        preds = []
        for pr in preds_per_label:
            preds.extend(pr)

        return {
            'Loss': loss/num, 
            'Accuracy': (accuracy_score(trues, preds), 
                [accuracy_score(t, p) 
                    for t, p in zip(trues_per_label, preds_per_label)]), 
            'Precision': (precision_score(trues, preds, average='macro', zero_division=np.nan), 
                [precision_score(t, p, average='macro', zero_division=np.nan) 
                    for t, p in zip(trues_per_label, preds_per_label)]), 
            'Recall': (recall_score(trues, preds, average='macro', zero_division=np.nan), 
                [recall_score(t, p, average='macro', zero_division=np.nan) 
                    for t, p in zip(trues_per_label, preds_per_label)]),
            'F1': (f1_score(trues, preds, average='macro', zero_division=np.nan), 
                [f1_score(t, p, average='macro', zero_division=np.nan) 
                    for t, p in zip(trues_per_label, preds_per_label)])
        }
    
    def train_few(self, train_loader):
        self.model.train()
        real_loader = self.load_real_dataset()
        real_iter = iter(real_loader)
        start = time.time()

        for e in range(self.args.client_epochs):
            for x, y in train_loader:
                x = x.to(self.args.device)
                y = y.to(self.args.device)
                output = self.model(x)
                loss = self.loss(output, y)
                # use real data
                try:
                    x_g, y_g = next(real_iter)
                except StopIteration:
                    real_iter = iter(real_loader)
                    x_g, y_g = next(real_iter)
                x_g = x_g.to(self.args.device)
                y_g = y_g.to(self.args.device)
                output_g = self.model(x_g)
                loss += self.loss(output_g, y_g)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            self.scheduler.step()

        print(f"\nClient training cost per iter.: {round(time.time()-start, 2)}s.")

    def train_metrics(self):
        train_loader = self.load_train_dataset()

        if self.args.task_type == 'syn':
            return self.train_zero_metrics(train_loader)
        elif self.args.task_type == 'mix':
            return self.train_few_metrics(train_loader)
        else:
            raise NotImplementedError 

    def train(self):
        train_loader = self.load_train_dataset()

        if self.args.client_retrain:
            self.model = get_model(self.args)
            self.opt = torch.optim.AdamW(self.model.head.parameters(), lr=self.args.client_learning_rate)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, self.args.client_epochs)

        if self.args.task_type == 'syn':
            self.train_zero(train_loader)
        elif self.args.task_type == 'mix':
            self.train_few(train_loader)
        else:
            raise NotImplementedError

    def avg_pair_psnr(self, real_per_label, gen_per_label):
        psnr_value_per_label = []
        for real_list, gen_list in zip(real_per_label, gen_per_label):
            psnr_value = 0
            cnt = len(real_list) * len(gen_list)
            for real_sample in real_list:
                for gen_sample in gen_list:
                    psnr_value += psnr(real_sample, gen_sample) / cnt
            psnr_value_per_label.append(psnr_value)
        return psnr_value_per_label

    def check_done(self):
        return sum(self.current_volume_per_label) >= self.args.num_labels * self.args.volume_per_label

    def run(self):
        self.train()
        self.callback()

    def callback(self):
        train_dataset_dir = os.path.join(self.args.dataset_dir, 'train', self.args.task)
        train_dataset_path = os.path.join(train_dataset_dir, f'{self.it-1}/dataset.pt')
        if os.path.exists(train_dataset_path):
            os.remove(train_dataset_path)


# https://github.com/jackfrued/Python-1/blob/master/analysis/compression_analysis/psnr.py
def psnr(original, contrast):
    mse = np.mean((original - contrast) ** 2) / 3
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    PSNR = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return PSNR