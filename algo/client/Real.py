import os
import torch
import time
from torch.utils.data import DataLoader
from algo.client.ClientBase import ClientBase
from utils.model import get_model 


class Client(ClientBase):
    def __init__(self, args):
        super().__init__(args)
        self.args.client_epochs_real = self.args.volume_per_label // self.args.real_volume_per_label * self.args.client_epochs


    def load_train_dataset(self):
        real_dataset_dir = os.path.join(self.args.dataset_dir, f'real/{self.args.real_volume_per_label}')
        real_dataset = torch.load(os.path.join(real_dataset_dir, self.args.client_dataset + '.pt'))
        return DataLoader(real_dataset, self.args.client_batch_size, drop_last=False, shuffle=True)
    
    def train(self):
        train_loader = self.load_train_dataset()
        if self.args.client_retrain:
            self.model = get_model(self.args)
            self.opt = torch.optim.AdamW(self.model.head.parameters(), lr=self.args.client_learning_rate)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, self.args.client_epochs)
        start = time.time()

        for e in range(self.args.client_epochs_real):
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
    
    def run(self):
        self.train()
        self.callback()