import torch
from thingsvision import get_extractor
from thingsvision.utils.storing import save_features
from thingsvision.utils.data import ImageDataset, DataLoader
import argparse
import os

from models_config import config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings_dir",
        default="/proj/rep-learning-robotics/users/x_nonra/gLocal/embeddings"
    )
    parser.add_argument(
        "--data_root",
        default="/proj/rep-learning-robotics/users/x_nonra/alignvis/data/things_eeg_2"
    )
    parser.add_argument("--dataset", type=str, default="things-eeg-2")
    parser.add_argument("--model_name", type=str, default="ViT-L-14")
    parser.add_argument("--split", type=str, default="train")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.embeddings_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    split = args.split
    if split == 'train':
        if args.dataset == 'things-eeg-2':
            data_root = os.path.join(args.data_root, 'training_images')
        elif args.dataset == 'things-meg':
            data_root = os.path.join(args.data_root, 'images_meg')
        else:
            raise ValueError("Invalid dataset")
    else:
        if args.dataset == 'things-eeg-2':
            data_root = os.path.join(args.data_root, 'test_images')
        elif args.dataset == 'things-meg':
            data_root = os.path.join(args.data_root, 'images_test_meg')
        else:
            raise ValueError("Invalid dataset")
    
    model_config = config[args.model_name]

    if 'model_parameters' in model_config.keys():
        extractor = get_extractor(
            model_name=model_config['model_name'],
            source=model_config['source'],
            device=device,
            pretrained=True,
            model_parameters=model_config['model_parameters'],
            )
    else:
        extractor = get_extractor(
            model_name=model_config['model_name'],
            source=model_config['source'],
            device=device,
            pretrained=True,
            )

    batch_size = 32

    dataset = ImageDataset(
        root=data_root,
        out_path=args.embeddings_dir,
        backend=extractor.get_backend(), # backend framework of model
        transforms=extractor.get_transformations(resize_dim=224, crop_dim=224) # set the input dimensionality to whichever values are required for your pretrained model
    )

    batches = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        backend=extractor.get_backend() # backend framework of model
    )

    features = extractor.extract_features(
        batches=batches,
        module_name=model_config['module_name'],
        flatten_acts=False,
        output_type="ndarray", # or "tensor" (only applicable to PyTorch models of which CLIP and DINO are ones!)
    )

    if "dino" in args.model_name.lower():
        features = features[:, 0, :].squeeze()
    print(features.shape)

    save_features(features, out_path=args.embeddings_dir, file_format='npy') # file_format can be set to "npy", "txt", "mat", "pt", or "hdf5"

    os.rename(os.path.join(args.embeddings_dir, f"features.npy"), os.path.join(args.embeddings_dir, f"{split}_gLocal_{args.model_name.lower()}_noalign.npy"))
    os.rename(os.path.join(args.embeddings_dir, f"file_names.txt"), os.path.join(args.embeddings_dir, f"filenames_{split}_gLocal_{args.model_name.lower()}_noalign.txt"))