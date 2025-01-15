import os
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings_dir",
        default="/proj/rep-learning-robotics/users/x_nonra/alignvis/data/things_eeg_img_embeddings/gLocal_test"
    )
    parser.add_argument(
        "--output_dir",
        default="/proj/rep-learning-robotics/users/x_nonra/alignvis/data/things_eeg_2/image_embeddings"
    )
    
    parser.add_argument("--model_name", type=str, default="OpenCLIP_ViT-L-14_laion2b_s32b_b82k")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument('--save_noalign', action='store_true', help='Save noalign embeddings')

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    model_name = args.model_name.lower()
    # Define the input file and new root path
    input_file = os.path.join(args.embeddings_dir, "filenames_"+args.split+"_gLocal_"+model_name+"_"+"noalign.txt")
    new_root_path = args.output_dir

    ndarrays_noalign = np.load(os.path.join(args.embeddings_dir, args.split+"_gLocal_"+model_name+"_"+"noalign.npy"))
    ndarrays_aligned = np.load(os.path.join(args.embeddings_dir, args.split+"_gLocal_"+model_name+".npy"))
    # Read the file line by line
    with open(input_file, "r") as file:
        lines = file.readlines()

    # Process each line
    for i, line in enumerate(lines):
        # Strip the newline character
        image_path = line.strip()
        
        # Extract the part of the path after "images"
        if "images/" in image_path:
            relative_path = image_path.split("images/", 1)[1]
        else: 
            relative_path = "images_test_meg/" + image_path  # MEG test filenames only include the image name and not the path
        
        # Replace .jpg with .npy
        npy_relative_path_noalign = relative_path.replace(".jpg", "_gLocal_"+model_name+"_noalign.npy")
        npy_relative_path_aligned = relative_path.replace(".jpg", "_gLocal_"+model_name+".npy")
        
        # Create the new path by joining the root path with the modified relative path
        npy_save_path_noalign = os.path.join(new_root_path, npy_relative_path_noalign)
        npy_save_path_aligned = os.path.join(new_root_path, npy_relative_path_aligned)
  
        if args.save_noalign:
            np.save(npy_save_path_noalign, np.expand_dims(ndarrays_noalign[i], axis=0))
        np.save(npy_save_path_aligned, np.expand_dims(ndarrays_aligned[i], axis=0))

        if i % 1000 == 0:
            print(f"Processed {i} images")

    print(f"Processed {len(lines)} images")
