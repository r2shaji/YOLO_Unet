import torch, json
import argparse
from train import Trainer
from test import Tester
from models.encoder import FeatureExtractor

def main():
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(description="Process command-line arguments.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file")
    parser.add_argument('--mode', type=str, required=True, help="train or test mode")
    
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    label_names = {int(key): value for key, value in config["label_names"].items()}
    config["label_names"] = label_names

    # Configuration
    embed_layers = [1,2,3,4,5]

    feature_extractor = FeatureExtractor(config,embed_layers)
    if args.mode == "train":
        trainer = Trainer(config, feature_extractor,config["num_epochs"])
        trainer.train()
    elif args.mode == "test":
        tester = Tester(config, feature_extractor)
        tester.test()
    else:
        print("Pass the right mode")

    return

if __name__=="__main__":
    main()
    

    