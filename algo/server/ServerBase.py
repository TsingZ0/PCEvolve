import glob
import os
import re
import time
import torch
import ujson
from utils.dataset import preprocess_image 
from utils.generator import get_generator 
from utils.prompts import *


class ServerBase(object):
    def __init__(self, args):
        self.client = args.Client(args)
        self.args = args
        self.current_volume_per_label = [0 for _ in range(args.num_labels)]
        self.done = False
        self.it = 0

        self.train_acc = []
        self.train_loss = []
        self.test_acc = []
        self.FID = []
        self.PSNR = []

        if not args.use_generated:
            self.base_prompt = base_prompt
            self.negative_prompt = negative_prompt
            self.Gen = get_generator(args)
            
            self.generated_dataset_dir = os.path.join(self.args.dataset_dir, 'generated', self.args.task)
            if not os.path.exists(self.generated_dataset_dir):
                os.makedirs(self.generated_dataset_dir)
            
        self.train_dataset_dir = os.path.join(self.args.dataset_dir, 'train', self.args.task)
        if not os.path.exists(self.train_dataset_dir):
            os.makedirs(self.train_dataset_dir)

    
    def send(self):
        self.client.it = self.it
        dataset = []
        for file_name in glob.glob(os.path.join(self.generated_dataset_dir, f'{self.it}/*.jpg')):
            img_tensor = preprocess_image(self.args, file_name)
            search = re.search(r"\[(.+)\]", file_name)
            if search is not None:
                label_name = search.group(1)
            else:
                label_name = ''
            label_id = torch.tensor(self.args.label_names.index(label_name))
            dataset.append((img_tensor.to(self.args.device), label_id.to(self.args.device)))
        print('\nGenerated volume per label: {:.4f}'.format(len(dataset) / self.args.num_labels))
        current_dir = os.path.join(self.train_dataset_dir, f'{self.it}')
        os.makedirs(current_dir)
        torch.save(dataset, os.path.join(current_dir, 'dataset.pt'))
        self.client.receive()

    def receive(self):
        self.current_volume_per_label, self.done = self.client.send()

    def get_prompt(self, label_name):
        prompt = self.base_prompt.format(
            DOMAIN=self.args.domain, 
            LABEL=label_name
        )[:self.args.prompt_max_length]
        return prompt

    def get_img(self, label_name):
        return None

    def generate(self):
        current_dir = os.path.join(self.generated_dataset_dir, f'{self.it}')
        os.makedirs(current_dir)
        image_urls_dict = {}
        for i, label_name in enumerate(self.args.label_names):
            offset = self.current_volume_per_label[i]
            while offset < self.args.volume_per_label:
                prompt = self.get_prompt(label_name)
                img = self.get_img(label_name)
                if img is None:
                    generated_images, image_urls = self.Gen(prompt, self.negative_prompt)
                else:
                    generated_images, image_urls = self.Gen(prompt, img, self.negative_prompt)
                result_images = generated_images[:self.args.volume_per_label - offset]
                result_urls = image_urls[:self.args.volume_per_label - offset]

                for idx, img in enumerate(result_images):
                    file_name = f'[{label_name}]-{offset + idx}.jpg'
                    img.save(os.path.join(current_dir, file_name))
                
                if len(result_urls) > 0:
                    for idx, url in enumerate(result_urls):
                        file_name = f'[{label_name}]-{offset + idx}.jpg'
                        image_urls_dict[file_name] = url

                offset += len(result_images)

        with open(os.path.join(current_dir, 'image_urls.json'), 'w') as f:
            ujson.dump(image_urls_dict, f)
                    

    def eval(self):
        train_metrics = self.client.train_metrics()
        eval_metrics = self.client.eval_metrics()

        print('\nTrain metrics:')
        if train_metrics:
            for metric, value in train_metrics.items():
                if metric == 'Loss':
                    print(metric, '=', f'{value:.4f}')
                else:
                    print(metric, '=', f'{value[0]:.4f}')
                    print(metric, 'per label =', *[f'{i}:{v:.4f}' for i, v in enumerate(value[1])])
            self.train_acc.append(train_metrics['Accuracy'][0])
            self.train_loss.append(train_metrics['Loss'])
            
        print('\nTest metrics:')
        for metric, value in eval_metrics.items():
            print(metric, '=', f'{value[0]:.4f}')
            print(metric, 'per label =', *[f'{i}:{v:.4f}' for i, v in enumerate(value[1])])
        self.test_acc.append(eval_metrics['Accuracy'][0])
        self.FID.append(eval_metrics['FID'][0])
        self.PSNR.append(eval_metrics['PSNR'][0])
    
    def check_done(self, lss):
        for ls in lss:
            find_top = len(ls) - torch.topk(torch.tensor(ls), 1).indices[0] > self.args.top_count
            if find_top:
                pass
            else:
                return False
        return True
    
    def callback(self):
        if not self.args.use_generated:
            del self.Gen
            
    def run(self):
        print(f"\n-------------Initial evaluation-------------")
        self.eval()
        start = time.time()
        print(f"\n-------------Iter. number: {0}-------------")

        if not self.args.use_generated:
            self.receive()
            self.generate()
            self.send()
        self.client.run()
        self.eval()

        print(f"\nTotal running cost per iter.: {round(time.time()-start, 2)}s.\n")

        self.callback()