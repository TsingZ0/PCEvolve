import copy
import os
import torch
import torch.nn.functional as F
from algo.client.ClientBase import ClientBase
from torch.utils.data import DataLoader
from utils.model import get_model
from collections import defaultdict


class Client(ClientBase):
    def __init__(self, args):
        super().__init__(args)
        
        self.real_vecs = [[] for _ in range(self.args.num_labels)]
        real_loader = self.load_real_dataset(batch_size=1)
        for x, y in real_loader:
            x = x.to(self.args.device)
            y = y.to(self.args.device)
            vecs = self.encoder(x).detach()
            for vec, yy in zip(vecs, y):
                yc = yy.item()
                self.real_vecs[yc].append(vec.detach().data)
        self.ref_imgs_prob = defaultdict(list)


    # send data back for extension, such as altering syn data
    def send(self, content=None):
        if content == 'real':
            real_data = self.load_real_dataset(is_raw=True)
            return copy.deepcopy(self.current_volume_per_label), self.done, real_data
        elif content == 'rated':
            if self.args.client_send_topk:
                self.select_topk()
            for label_id, imps in self.ref_imgs_prob.items():
                imp_sum = sum(imps)
                assert imp_sum > 0, 'at least one sample probability > 0'
                imps_new = [imp/imp_sum for imp in imps]
                self.ref_imgs_prob[label_id] = imps_new
            return copy.deepcopy(self.current_volume_per_label), self.done, copy.deepcopy(self.ref_imgs_prob)
        else:
            return copy.deepcopy(self.current_volume_per_label), self.done
        
    def load_filtered_dataset(self, is_raw=False, batch_size=None):
        filtered_dataset_dir = os.path.join(self.args.dataset_dir, 'train', self.args.task)
        try:
            filtered_dataset = torch.load(os.path.join(filtered_dataset_dir, f'{self.it}/filtered_dataset.pt'))
            if is_raw:
                return filtered_dataset
            else:
                if batch_size is None:
                    batch_size = self.args.client_batch_size
                filtered_dataset_new = []
                for data in filtered_dataset:
                    for x, y in data:
                        filtered_dataset_new.append((x, y))
                return DataLoader(filtered_dataset_new, batch_size, drop_last=False, shuffle=True)
        except (FileNotFoundError, ValueError):
            return None
        
    def train_metrics(self):
        if self.args.client_use_filtered:
            train_loader = self.load_filtered_dataset()
        else:
            train_loader = self.load_train_dataset()

        if self.args.task_type == 'syn':
            return self.train_zero_metrics(train_loader)
        elif self.args.task_type == 'mix':
            return self.train_few_metrics(train_loader)
        else:
            raise NotImplementedError 

    def train(self):
        if self.args.client_use_filtered:
            train_loader = self.load_filtered_dataset()
        else:
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

    def selector(self, filtered_dataset):
        raise NotImplementedError("Please Implement the selector method")
    
    def calculate_dist(self, vec_a, vec_b):
        dist = torch.norm(vec_a - vec_b, p=2) # L2
        return dist.detach().item()
    
    def select_topk(self):
        for label_id, imps in self.ref_imgs_prob.items():
            top_value = torch.topk(torch.tensor(imps), 
                                   self.args.client_topk_per_label).values[-1].item()
            imps_new = [imp if imp >= top_value else 0 for imp in imps]
            self.ref_imgs_prob[label_id] = imps_new

    def run(self):
        filtered_dataset_dir = os.path.join(self.args.dataset_dir, 'train', self.args.task)
        if self.args.client_accumulate_filter:
            try:
                filtered_dataset = torch.load(os.path.join(filtered_dataset_dir, f'{self.it-1}/filtered_dataset.pt'))
            except FileNotFoundError:
                filtered_dataset = [[] for _ in range(self.args.num_labels)]
        else:
            filtered_dataset = [[] for _ in range(self.args.num_labels)]

        filtered_dataset, last_yc = self.selector(filtered_dataset)
        if sum(self.current_volume_per_label) % self.args.client_batch_size == 1:
            filtered_dataset[last_yc] = filtered_dataset[last_yc][:-1]
            self.current_volume_per_label[last_yc] -= 1
        print('Client filtered size:', self.current_volume_per_label)
        torch.save(filtered_dataset, os.path.join(filtered_dataset_dir, f'{self.it}/filtered_dataset.pt'))

        if not self.args.client_accumulate_filter:
            self.current_volume_per_label = [0 for _ in range(self.args.num_labels)]
            self.done = False

        self.train()
        self.callback()

    def callback(self):
        train_dataset_dir = os.path.join(self.args.dataset_dir, 'train', self.args.task)
        train_dataset_path = os.path.join(train_dataset_dir, f'{self.it-1}/dataset.pt')
        if os.path.exists(train_dataset_path):
            os.remove(train_dataset_path)
            
        filtered_dataset_dir = os.path.join(self.args.dataset_dir, 'train', self.args.task)
        filtered_dataset_path = os.path.join(filtered_dataset_dir, f'{self.it-1}/filtered_dataset.pt')
        if os.path.exists(filtered_dataset_path):
            os.remove(filtered_dataset_path)