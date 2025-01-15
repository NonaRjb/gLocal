import argparse
import os
import warnings

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from downstream.retrieval import CLIP_MODEL_MAPPING, CLIP_MODELS, embeddings
from downstream.retrieval.eval import evaluate
from downstream.utils import THINGSFeatureTransform


def gen_transformed_embeddings(
        embeddings_dir: str,
        data_root: str,
        transform_path: str,
        update_transforms: bool = False,
        concat_weight=None,
        model_name="ViT-L-14",
        split="train",
):
    all_results = []

    try:
        embeddings = np.load(os.path.join(embeddings_dir, f"{split}_gLocal_{model_name.lower()}_noalign.npy"))
    except FileNotFoundError:
        warnings.warn(
            message=f"\nCould not find embedding file for {model_name}. Skipping current evaluation and continuing with next CLIP model...\n",
            category=UserWarning,
            stacklevel=1,
        )
    img_embedding = torch.from_numpy(embeddings)
    
    if update_transforms:
        # model_key = CLIP_MODEL_MAPPING[model_name]
        things_feature_transform = THINGSFeatureTransform(
            source="custom",
            # model_name=model_key,
            module="penultimate",
            path_to_transform=transform_path,
        )
        image_transformed = torch.tensor(
            things_feature_transform.transform_features(img_embedding)
        )
        
        np.save(
            os.path.join(embeddings_dir, f"{split}_gLocal_{model_name.lower()}.npy"),
            image_transformed.detach().numpy()
        )
        print(f"\nTransformed embeddings saved to {embeddings_dir}\n")
    else:
        image_transformed = torch.from_numpy(embeddings)
    
    if concat_weight is not None:
        print("\nUsing weighted concat with", concat_weight)
        print(f"Shape: {image_transformed.shape}\n")
        image_transformed = torch.cat(
            img_embedding * (1 - concat_weight),
            image_transformed * concat_weight,
            dim=1,
        )
        np.save(
            os.path.join(embeddings_dir, f"{split}_gLocal_{model_name.lower()}.npy"),
            image_transformed.detach().numpy()
        )
    
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings_dir", 
        default="/proj/rep-learning-robotics/users/x_nonra/gLocal/embeddings"
        )
    parser.add_argument(
        "--data_root", 
        default="/proj/rep-learning-robotics/users/x_nonra/alignvis/data/things_eeg_2"
        )
    parser.add_argument(
        "--transform_path",
        default="/proj/rep-learning-robotics/users/x_nonra/gLocal/transforms/clip_ViT-L/14/visual/transform.npz",
        )
    parser.add_argument("--model_name", default="ViT-L-14")
    parser.add_argument("--update_transforms", action="store_true")
    parser.add_argument(
        "--concat-weight",
        type=float,
        default=None,
        help="Off by default. Set to a weighing factor to concat embeddings and weigh by the <factor>",
        )
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()

    os.makedirs(args.embeddings_dir, exist_ok=True)

    if os.path.exists(os.path.join(args.embeddings_dir, f"train_gLocal_{args.model_name.lower()}_noalign.npy")):
        print(f"\nEmbeddings already exist for {args.model_name}. Skipping computation...\n")
    
    else: 
        for split in ["train", "test"]:
            print(f"\nComputing {split} embeddings for non-human-aligned {args.model_name}")
            embeddings.compute_things_image_embeddings(
                embeddings_folder=args.embeddings_dir,
                device=args.device,
                data_root=args.data_root,
                model_name=args.model_name,
                split=split,
            )

    for split in ["train", "test"]:
        print(f"\nTransforming {split} embeddings for human-aligned {args.model_name}")
        gen_transformed_embeddings(
            embeddings_dir=args.embeddings_dir,
            update_transforms=args.update_transforms,
            concat_weight=args.concat_weight,
            data_root=args.data_root,
            transform_path=args.transform_path,
            model_name=args.model_name,
            split=split,
        )
    


