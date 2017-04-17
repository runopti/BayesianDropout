import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="trained_models")
parser.add_argument("--model", type=str, default="models/mlp")
parser.add_argument("--dataset_dir", type=str, default="/Users/yutaro/GoogleDrive/github/datasets/")
parser.add_argument("--seed", type=int, default=0)

args = parser.parse_args()

