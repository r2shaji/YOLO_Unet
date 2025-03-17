from torchvision import datasets, transforms
import torch.nn as nn
from ultralytics import YOLO
import torch
import glob
import numpy as np
from PIL import Image
import os
from dataset import xywhn_to_xyxy, load_label, transform_image
import util

class FeatureExtractor():

    def __init__(self, config, embed_layers=[1,2,3,4,5]):
        self.model = YOLO(config["yolo_path"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.label_folder = config["label_folder"]
        self.label_names = config["label_names"]
        self.embed_layers = embed_layers
        self.sharp_image_paths = sorted(
            glob.glob(os.path.join(config["sharp_image_folder"], "*.jpg"))
        )
        self.sharp_image_folder = config["sharp_image_folder"]
        self.blur_image_paths = sorted(
            glob.glob(os.path.join(config["blur_image_folder"], "*.jpg"))
        )

    def crop_features(self, feature_map, bbox):

        _, _, H, W = feature_map.shape
        xmin, ymin, xmax, ymax = xywhn_to_xyxy(bbox, H, W)
        cropped = feature_map[:, :, ymin:ymax, xmin:xmax]
        ch, cw = cropped.shape[2], cropped.shape[3]
        if cw == 0 or ch == 0:
            if cw == 0:
                xmin = max(xmin - 1, 0)
                xmax = min(xmax + 1, W)
            if ch == 0:
                ymin = max(ymin - 1, 0)
                ymax = min(ymax + 1, H)
            cropped = feature_map[:, :, ymin:ymax, xmin:xmax]

        return cropped

    def extract_image_features(self, image, ground_truth):

        true_boxes = ground_truth["sorted_boxes_xywhn"]
        results = self.model.predict(image, embed=self.embed_layers)
        cropped_char_features = [[] for _ in range(len(true_boxes))]
        k = 0

        for diff_layer_feature in results:
            k += 1
            # feature_visualization(diff_layer_feature, im_ht, im_wid, true_boxes, f"{image_name}_stage_{k}")
            for i, bbox in enumerate(true_boxes):
                label = self.label_names[ground_truth["sorted_labels"][i].item()]
                # plot_image_name = f'feature_crops_1_6_15_21/{image_name}_{i}_{label}.png'
                cropped_feats = self.crop_features(diff_layer_feature, bbox)
                # plot_and_save_cropped_feature_map(cropped_feats, plot_image_name)
                print("cropped_feats before pooling shape", cropped_feats.shape)
                cropped_feats = nn.functional.adaptive_avg_pool2d(cropped_feats, (1, 1)).squeeze(-1).squeeze(-1)
                print("cropped_feats after pooling shape", cropped_feats.shape)

                if torch.isnan(cropped_feats).any():
                    print("Tensor has NaN values.", cropped_feats)
                print("label", label)
                cropped_char_features[i].append(cropped_feats)

            del diff_layer_feature
            torch.cuda.empty_cache()

        true_labels = []
        for i, char_embeddings in enumerate(cropped_char_features):
            cropped_char_features[i] = torch.unbind(torch.cat(char_embeddings, 1), dim=0)[0].squeeze(0)
            print("cropped_feats", cropped_char_features[i].shape)
            true_labels.append(ground_truth["sorted_labels"][i].item())

        return cropped_char_features, true_labels

    def extract_image_features_classification(self):

        all_features = []
        true_labels = []
        for image_path in self.sharp_image_paths:
            image_name = os.path.basename(image_path).split(".")[0]
            ground_truth = load_label(image_path, self.label_folder)
            image_tensor = util.get_image_tensor(image_path)
            image_features, image_gt = self.extract_image_features(image_tensor, ground_truth)
            all_features.extend(image_features)
            true_labels.extend(image_gt)

        X_tensor = torch.stack(all_features)
        y_tensor = torch.tensor(true_labels, dtype=torch.long)

        return X_tensor, y_tensor
