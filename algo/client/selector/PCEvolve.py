import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from algo.client.selector.SelectorBase import Client as Selector
from torch.utils.data import DataLoader
from collections import defaultdict, Counter


class Client(Selector):
    def __init__(self, args):
        super().__init__(args)

        self.sensitivity = 1
        self.CEloss = nn.CrossEntropyLoss()
        self.get_proto()


    @torch.no_grad()
    def selector(self, filtered_dataset):
        self.ref_imgs_prob = defaultdict(list)
        self.model.eval()

        # Exponential mechanism param process
        Ks = [
            min(1, self.args.volume_per_label - self.current_volume_per_label[yc])
            for yc in range(self.args.num_labels)
        ]
        self.epsilon = self.args.epsilon / sum(Ks)

        # collect scores
        train_loader = self.load_train_dataset(is_shuffle=False, batch_size=1)
        score_list = [[] for _ in range(self.args.num_labels)]
        for x, y in train_loader:
            x = x.to(self.args.device)
            y = y.to(self.args.device)
            vecs = self.encoder(x).detach()
            for vec, yy in zip(vecs, y):
                yc = yy.item()
                score = self.get_score(vec, yc)
                score_list[yc].append(score)

        # adjust scores
        exp_list = [[] for _ in range(self.args.num_labels)]
        for yc in range(self.args.num_labels):
            score_min = float('inf')
            score_max = - float('inf')
            for score in score_list[yc]:
                if score > - float('inf'):
                    score_min = min(score_min, score)
                    score_max = max(score_max, score)

            if score_min < float('inf'):
                for score in score_list[yc]:
                    if score_max == score_min: # for debugging
                        score = 0
                    else:
                        score = - (score_max - score) / (score_max - score_min) # normalize to [-1,0]
                    score_new = np.exp(score * self.args.tau).item() # [0,1]
                    exp_value = np.exp(self.epsilon * score_new / (2*self.sensitivity)).item()
                    exp_list[yc].append(exp_value)
            else:
                exp_list[yc] = [1 for _ in score_list[yc]]
                print(f'Random selection for class {yc}.')

        # selector
        exp_prob = [np.array(exps) / sum(exps) for exps in exp_list]
        sampled_idxs = [[] for _ in range(self.args.num_labels)]
        for yc in range(self.args.num_labels):
            for _ in range(Ks[yc]):
                sampled_idx = np.random.choice(len(exp_prob[yc]), 1, p=exp_prob[yc])[0].item()
                sampled_idxs[yc].append(sampled_idx)
        counters = [Counter(idxs) for idxs in sampled_idxs]
        sampled_idxs = [list(counter.keys()) for counter in counters]
        print('Client idx count:', counters)
        print('Client rated idx:', sampled_idxs)
        
        # get filtered_dataset
        last_yc = None
        idxs = [0 for _ in range(self.args.num_labels)]
        for x, y in train_loader:
            for xx, yy in zip(x, y):
                yc = yy.item()
                if self.current_volume_per_label[yc] < self.args.volume_per_label and idxs[yc] in sampled_idxs[yc]:
                    filtered_dataset[yc].append((xx, yy))
                    self.current_volume_per_label[yc] += 1
                    last_yc = yc
                    imp = counters[yc][idxs[yc]]
                    self.ref_imgs_prob[yc].append(imp)
                else:
                    self.ref_imgs_prob[yc].append(0)
                idxs[yc] += 1
            if self.check_done():
                self.done = True
                break
        return filtered_dataset, last_yc


    def get_score(self, vec, yc):
        imps = []
        for yy, proto in enumerate(self.protos):
            dist = self.calculate_dist(vec, proto)
            if yy == yc:
                imp = - (dist)
                score = - dist
            else:
                imp = - dist
            imps.append(imp)
        imps = torch.tensor(imps)
        softmax_values = F.softmax(imps, dim=0).cpu().numpy().tolist()
        if softmax_values[yc] < max(softmax_values):
            score = - float('inf')
        return score # (-inf, 0]

    def get_proto(self):
        protos = [0 for _ in range(self.args.num_labels)]
        for i, vec_list in enumerate(self.real_vecs):
            if len(vec_list) > 1:
                proto = 0 * vec_list[0].data
                for vec in vec_list:
                    proto += vec.data
                protos[i] = proto / len(vec_list)
            else:
                protos[i] = vec_list[0]
        self.protos = protos

