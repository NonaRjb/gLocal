from torchvision.datasets import VisionDataset
import pandas as pd
from PIL import Image
import torch
import os
import open_clip
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import numpy as np


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class Flickr30kImages(VisionDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_path = os.path.join(self.root, 'flickr30k_images')
        df = pd.read_csv(os.path.join(self.root, 'results.csv'), sep="|")
        self.files = df.image_name.unique()

    def __getitem__(self, index: int):
        file = self.files[index]
        path = os.path.join(self.img_path, file)
        sample = pil_loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample

    def __len__(self):
        return len(self.files)


class Flickr30kCaptions(Dataset):

    def __init__(self, root):
        super().__init__()
        self.root = root
        df = pd.read_csv(os.path.join(self.root, 'results.csv'), sep="|")
        caption_key = ' comment'
        self.captions = df[caption_key]
        self.captions[self.captions.isna()] = ' '

    def __getitem__(self, index: int):
        caption = self.captions[index]
        return caption

    def __len__(self):
        return len(self.captions)


def compute_things_image_embeddings(model_name, data_root, embeddings_folder, split='train', device='cuda'):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='openai')
    model = model.to(device)
    img_parent_dir  = data_root
    img_metadata = np.load(os.path.join(img_parent_dir, 'image_metadata.npy'), allow_pickle=True).item()
    total_images = len(img_metadata[f'{split}_img_files'])
    print(f"Split = {split}")
    embeddings = []
    for item in range(total_images):
        if split == 'train':
            img_file = os.path.join(img_parent_dir, 'training_images', 
                            img_metadata['train_img_concepts'][item], img_metadata['train_img_files'][item])
        else:
            img_file = os.path.join(img_parent_dir, 'test_images', 
                            img_metadata['test_img_concepts'][item], img_metadata['test_img_files'][item])
        img = pil_loader(img_file)
        img = preprocess(img).to(device)
        with torch.no_grad():
            e = model.encode_image(img.unsqueeze(0)).detach().cpu().numpy()
        embeddings.append(e)
        # np.save(img_file.replace(".jpg", f"_gLocal_{model_name.lower()}_noalign.npy"), e)
        if item % 1000 == 0:
            print(f"{item} items out of {total_images} done")
            print(f"e.shape = {e.shape}")
    
    embeddings = np.array(embeddings).squeeze()
    print(f"embeddings.shape = {embeddings.shape}")
    np.save(os.path.join(embeddings_folder, f'{split}_{model_name.lower()}_noalign.npy'), embeddings)


def compute_image_embeddings(model, preprocess, data_root="flickr30k_images", device='cuda'):
    dataset = Flickr30kImages(root=data_root, transform=preprocess)
    loader = DataLoader(dataset, batch_size=32)
    embeddings = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for x in tqdm(loader):
            x = x.to(device)
            image_features = model.encode_image(x)
            embeddings.append(image_features.cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings


def compute_text_embeddings(model, tokenizer, data_root="flickr30k_images", device='cuda'):
    dataset = Flickr30kCaptions(root=data_root)
    loader = DataLoader(dataset, batch_size=32, num_workers=4)
    embeddings = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for x in tqdm(loader):
            x = tokenizer(x)
            x = x.to(device)
            text_features = model.encode_text(x)
            embeddings.append(text_features.cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings


def compute_embeddings(model_name, data_root, embeddings_folder, device='cuda'):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='openai')
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    image_embeddings = compute_image_embeddings(model=model, preprocess=preprocess,
                                                data_root=data_root, device=device)
    text_embeddings = compute_text_embeddings(model=model, tokenizer=tokenizer, data_root=data_root,
                                              device=device)
    np.savez(os.path.join(embeddings_folder, f'{model_name}.npz'),
             images=image_embeddings, text=text_embeddings)
