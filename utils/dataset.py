import os
import pandas as pd
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import ssl
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ssl._create_default_https_context = ssl._create_stdlib_context

# mean=[0.485, 0.456, 0.406]
# std=[0.229, 0.224, 0.225]
# normalize = transforms.Normalize(mean=mean, std=std)
# inv_normalize = transforms.Normalize(
#     mean=[-m/s for m, s in zip(mean, std)],
#     std=[1/s for s in std]
# )


def select_data(any_set, start, end):
    any_loader = DataLoader(any_set, len(any_set), shuffle=False)
    data, label = next(iter(any_loader))
    any_set_data = data.cpu().detach().numpy()
    any_set_label = label.cpu().detach().numpy()

    label_ids = set(any_set_label)
    any_set_data_used = None
    any_set_label_used = None
    for label_id in label_ids:
        if any_set_data_used is None:
            any_set_data_used = any_set_data[any_set_label == label_id][start:end]
            any_set_label_used = any_set_label[any_set_label == label_id][start:end]
        else:
            any_set_data_used = np.append(
                any_set_data_used, 
                any_set_data[any_set_label == label_id][start:end], axis=0
            )
            any_set_label_used = np.append(
                any_set_label_used, 
                any_set_label[any_set_label == label_id][start:end], axis=0
            )
    any_set = list(zip(any_set_data_used, any_set_label_used))
    return any_set


def get_real_data(args):
    real_exist, test_exist = False, False
    real_dataset_dir = os.path.join(args.dataset_dir, f'real/{args.real_volume_per_label}')
    test_dataset_dir = os.path.join(args.dataset_dir, 'test')
    real_file_path = os.path.join(real_dataset_dir, args.client_dataset + '.pt')
    test_file_path = os.path.join(test_dataset_dir, args.client_dataset + '.pt')
    label_file_path = os.path.join(test_dataset_dir, args.client_dataset + '-label_names.pt')
    task_file_path = os.path.join(test_dataset_dir, args.client_dataset + '-domain.pt')
    if args.real_volume_per_label == 0:
        real_exist = True
    else:
        if not os.path.exists(real_dataset_dir):
            os.makedirs(real_dataset_dir)
        elif os.path.exists(real_file_path):
            real_exist = True
    if not os.path.exists(test_dataset_dir):
        os.makedirs(test_dataset_dir)
    elif os.path.exists(test_file_path):
        test_exist = True

    def get_transform(img_size):
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        return transform

    if not real_exist or not test_exist:
        if args.client_dataset == 'Camelyon17':
            from wilds import get_dataset
            dataset = get_dataset(
                dataset='camelyon17', 
                root_dir=os.path.join(args.dataset_dir, 'rawdata'), 
                download=True, 
            )
            transform=transforms.ToTensor()
            real_set = [(transform(x), y) for x, y, _ in dataset.get_subset('train')]
            test_set = [(transform(x), y) for x, y, _ in dataset.get_subset('test')]
            H = test_set[0][0].shape[-2]
            W = test_set[0][0].shape[-1]
            args.img_size = min(min(H, W), args.image_max_size)
            domain = 'histological lymph node section'
            label_names = ['', 'breast cancer with a tumor tissue']
        elif args.client_dataset == 'kvasir':
            # https://datasets.simula.no/kvasir/
            data_dir = 'dataset/rawdata/kvasir-dataset-v2/'
            label_names = ['esophagitis', 'polyps', 'ulcerative-colitis']
            file_names = []
            labels = []
            for dir in os.listdir(data_dir):
                if dir in label_names:
                    label = label_names.index(dir)
                    for file_name in os.listdir(os.path.join(data_dir, dir)):
                        file_names.append(os.path.join(dir, file_name))
                        labels.append(label)
            df = pd.DataFrame({'file_name': file_names, 'class': labels})

            dataset = ImageDataset(df, data_dir, transforms.ToTensor())
            any_loader = DataLoader(dataset)
            H = next(iter(any_loader))[0].shape[-2]
            W = next(iter(any_loader))[0].shape[-1]
            args.img_size = min(min(H, W), args.image_max_size)

            transform = transforms.Compose(
                [transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()])
            dataset = ImageDataset(df, data_dir, transform)
            full_len_per_label = len(dataset) // len(label_names)
            test_index = int(full_len_per_label * args.test_ratio)
            test_set = select_data(dataset, 0, test_index)
            real_set = select_data(dataset, test_index, full_len_per_label)
            domain = 'pathological damage in mucosa of gastrointestinal tract'
        elif args.client_dataset == 'COVIDx':
            # https://www.kaggle.com/datasets/andyczhao/covidx-cxr2
            data_dir = 'dataset/rawdata/COVIDx/'
            val_df = pd.read_csv(data_dir + 'val.txt', sep=" ", header=None)
            val_df.columns=['patient_id', 'file_name', 'class', 'data_source']
            val_df.drop(columns=['patient_id', 'data_source'])
            val_df['class'] = val_df['class'] == 'positive'
            val_df['class'] = val_df['class'].astype(int)

            test_df = pd.read_csv(data_dir + 'test.txt', sep=" ", header=None)
            test_df.columns=['patient_id', 'file_name', 'class', 'data_source']
            test_df.drop(columns=['patient_id', 'data_source'])
            test_df['class'] = test_df['class'] == 'positive'
            test_df['class'] = test_df['class'].astype(int)

            any_set = ImageDataset(val_df, data_dir+'val/', transforms.ToTensor())
            any_loader = DataLoader(any_set)
            H = next(iter(any_loader))[0].shape[-2]
            W = next(iter(any_loader))[0].shape[-1]
            args.img_size = min(min(H, W), args.image_max_size)

            transform = transforms.Compose(
                [transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()])
            real_set = ImageDataset(val_df, data_dir+'val/', transform)
            real_loader = DataLoader(real_set)
            real_set = [(x[0], y[0]) for x, y in real_loader]
            test_set = ImageDataset(test_df, data_dir+'test/', transform)
            test_loader = DataLoader(test_set)
            test_set = [(x[0], y[0]) for x, y in test_loader]
            domain = 'chest radiography (X-ray)'
            label_names = ['', 'COVID-19 pneumonia']
        elif args.client_dataset == 'MVTecADLeather':
            # https://www.mvtec.com/company/research/datasets/mvtec-ad
            data_dir = 'dataset/rawdata/MVTecADLeather/'
            dir_names = ['good', 'cut', 'glue']
            # only 19 images in each class
            file_names = []
            labels = []
            for dir in os.listdir(data_dir):
                if dir in dir_names:
                    label = dir_names.index(dir)
                    for file_name in os.listdir(os.path.join(data_dir, dir)):
                        file_names.append(os.path.join(dir, file_name))
                        labels.append(label)
            df = pd.DataFrame({'file_name': file_names, 'class': labels})

            dataset = ImageDataset(df, data_dir, transforms.ToTensor())
            any_loader = DataLoader(dataset)
            H = next(iter(any_loader))[0].shape[-2]
            W = next(iter(any_loader))[0].shape[-1]
            args.img_size = min(min(H, W), args.image_max_size)

            transform = transforms.Compose(
                [transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()])
            dataset = ImageDataset(df, data_dir, transform)
            label_names = ['', 'cut defect', 'droplet defect']
            full_len_per_label = len(dataset) // len(label_names)
            test_index = int(full_len_per_label * args.test_ratio)
            test_set = select_data(dataset, 0, test_index)
            real_set = select_data(dataset, test_index, full_len_per_label)
            domain = 'leather texture'
        else:
            raise NotImplemented
            
        real_set = select_data(real_set, 0, args.real_volume_per_label)
        print(f'Real and test datasets created.')
        if not os.path.exists(real_file_path):
            torch.save(real_set, real_file_path)
        if not os.path.exists(test_file_path):
            torch.save(test_set, test_file_path)
        if not os.path.exists(label_file_path):
            torch.save(label_names, label_file_path)
        if not os.path.exists(task_file_path):
            torch.save(domain, task_file_path)
    else:
        print('Real and test datasets already exist.')

    test_set = torch.load(test_file_path)
    if args.real_volume_per_label == 0:
        print(f'Test set size: {len(test_set)}.')
    else:
        real_set = torch.load(real_file_path)
        print(f'Real set size: {len(real_set)}, test set size: {len(test_set)}.')

    label_names = torch.load(label_file_path)
    domain = torch.load(task_file_path)
    args.label_names = label_names
    args.num_labels = len(label_names)
    args.domain = domain
    print(f'Labels: {args.label_names}')
    print(f'Number of labels: {args.num_labels}')
    print(f'Client domain: {args.domain}')
    H = test_set[0][0].shape[-2]
    W = test_set[0][0].shape[-1]
    args.img_size = min(H, W)
    print(f'Client image size: {H}x{W}')


def preprocess_image(args, image_path):
    # Load image
    img = Image.open(image_path)

    # Resize image
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])

    # Apply transformations
    img_tensor = transform(img)

    return img_tensor


class ImageDataset(Dataset):
    def __init__(self, dataframe, image_folder, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing file names
            image_folder (str): Path to the folder containing the images
            transform (callable, optional): Optional transform to be applied to the image
        """
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get the file name from the DataFrame
        img_name = self.dataframe.iloc[idx]['file_name']
        img_label = self.dataframe.iloc[idx]['class']
        img_path = os.path.join(self.image_folder, img_name)
        
        # Load the image using PIL
        image = Image.open(img_path).convert('RGB')  # Ensure RGB if not grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_label