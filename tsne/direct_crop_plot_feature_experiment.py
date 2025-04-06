import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.manifold import TSNE
import cv2
import torch
import torch.nn as nn
from matplotlib.cm import get_cmap


# Possible YOLO layers to extract
embed_layer_configs = [
    [1,2,3,4,5],
    [1, 2, 4, 10, 13, 16],
    [8, 9, 10, 13, 16],
    [16,17, 19,20],
    [9,10,16,19, 20]
]

# t-SNE hyperparameter sets
tsne_configs = [
    {"perplexity": 30, "n_iter": 3000},
    {"perplexity": 40, "n_iter": 3000},
    {"perplexity": 50, "n_iter": 3000}
]


MODEL_PATH = r"D:\Users\r2shaji\Downloads\best_copy.pt"

IMAGE_FOLDER = r"D:\Users\r2shaji\Downloads\char_low_noise\images\val"
LABEL_FOLDER = r"D:\Users\r2shaji\Downloads\char_low_noise\labels\val"

model = YOLO(MODEL_PATH)

image_paths = sorted(
    glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")) +
    glob.glob(os.path.join(IMAGE_FOLDER, "*.png")) +
    glob.glob(os.path.join(IMAGE_FOLDER, "*.jpeg"))
)
# label_names = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: '#'}
label_names = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f', 16: 'g', 17: 'h', 18: 'i', 19: 'j', 20: 'k', 21: 'l', 22: 'm', 23: 'n', 24: 'o', 25: 'p', 26: 'q', 27: 'r', 28: 's', 29: 't', 30: 'u', 31: 'v', 32: 'w', 33: 'x', 34: 'y', 35: 'z', 36: 'A', 37: 'B', 38: 'C', 39: 'D', 40: 'E', 41: 'F', 42: 'G', 43: 'H', 44: 'I', 45: 'J', 46: 'K', 47: 'L', 48: 'M', 49: 'N', 50: 'O', 51: 'P', 52: 'Q', 53: 'R', 54: 'S', 55: 'T', 56: 'U', 57: 'V', 58: 'W', 59: 'X', 60: 'Y', 61: 'Z'}
subset = []

def load_label(image_path):
        label_file_name = os.path.basename(image_path).replace('.jpg', '.txt')
        label_file_path = os.path.join(LABEL_FOLDER,label_file_name)
        with open(label_file_path, 'r') as f:
            lines = f.readlines()
        lines= [line.rstrip() for line in lines]
        lines= [line.split() for line in lines]
        lines = np.array(lines).astype(float)
        lines = torch.from_numpy(lines)
        lines = sorted(lines, key=lambda x: x[1])
        sorted_labels = torch.tensor([t[0] for t in lines], dtype=torch.float)
        sorted_labels = sorted_labels.long()
        sorted_boxes = [t[1:] for t in lines]
        plate_info = { "sorted_labels": sorted_labels, "sorted_boxes_xywhn":sorted_boxes}
        return plate_info

def xywhn_to_xyxy(detection, height, width):
    cx, cy, w, h = detection
        
    x1 = int((cx - w / 2) * width)
    y1 = int((cy - h / 2) * height)
    x2 = int((cx + w / 2) * width)
    y2 = int((cy + h / 2) * height)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    return x1, y1, x2, y2


def crop_features(feature_map, bbox):
    # if isinstance(feature_map, tuple):
    #     print("feature_map", feature_map[0].shape)
    #     print("length", feature_map[1].shape)
    # else:
    #     print("feature_map.shape",feature_map.shape)


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


def extract_embeddings_and_labels(image_path, embed_layers):

    detection_results = load_label(image_path)
    
    if not detection_results["sorted_boxes_xywhn"]:
        return [] 
    
    extracted = []

    true_boxes = detection_results["sorted_boxes_xywhn"]
    with torch.no_grad():
        results = model.predict(image_path, embed=embed_layers)
    cropped_char_features = [[] for _ in range(len(true_boxes))]

    k=0
    for diff_layer_feature in results:
        print("k",k)
        k+=1
        # feature_visualization(diff_layer_feature, im_ht, im_wid, true_boxes, f"{image_name}_stage_{k}")
        for i, bbox in enumerate(true_boxes):
            # label = label_names[detection_results["sorted_labels"][i].item()]
            # plot_image_name = f'feature_crops_1_6_15_21/{image_name}_{i}_{label}.png'
            cropped_feats = crop_features(diff_layer_feature, bbox)
            # plot_and_save_cropped_feature_map(cropped_feats, plot_image_name)
            # print("cropped_feats before pooling shape", cropped_feats.shape)
            cropped_feats = nn.functional.adaptive_avg_pool2d(cropped_feats, (1, 1)).squeeze(-1).squeeze(-1)
            # print("cropped_feats after pooling shape", cropped_feats.shape)

            if torch.isnan(cropped_feats).any():
                print("Tensor has NaN values.", cropped_feats)
            # print("label", label)
            cropped_char_features[i].append(cropped_feats)

        del diff_layer_feature
        torch.cuda.empty_cache()

    for i, char_embeddings in enumerate(cropped_char_features):
        cropped_char_features[i] = torch.unbind(torch.cat(char_embeddings, 1), dim=0)[0].squeeze(0)
        lbl = detection_results["sorted_labels"][i].item()
        if int(lbl)>9 and int(lbl)<36:
            extracted.append((cropped_char_features[i], lbl))

    return extracted


for embed_layers in embed_layer_configs:
    os.makedirs(f"plot_exp_direct_crop_small_letters/{embed_layers}",exist_ok=True)
    print(f"Extracting embeddings from layers: {embed_layers}")

    embeddings = []
    labels = []
    for path in image_paths:
        extracted = extract_embeddings_and_labels(path, embed_layers)
        for emb, lbl in extracted:
            embeddings.append(emb)
            labels.append(lbl)

    print("length of labels",len(labels))
    
    if len(embeddings) == 0:
        print("No detections found with this layer config. Skipping.")
        continue
    features_all = np.vstack(embeddings)
    
    for cfg in tsne_configs:
        perplex = cfg["perplexity"]
        n_iter = cfg["n_iter"]
        print(f"Running t-SNE with perplexity={perplex}")
        
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplex,
            n_iter=n_iter
        )
        tsne_results = tsne.fit_transform(features_all)
        
        unique_labels = sorted(set(labels))
        cmap = get_cmap("tab20", len(unique_labels))
        label_to_color = {label: cmap(i) for i, label in enumerate(unique_labels)}
        
        plt.figure(figsize=(10, 8))
        for i, lbl in enumerate(labels):
            plt.scatter(
                tsne_results[i, 0],
                tsne_results[i, 1],
                color=label_to_color[lbl],
                alpha=0.6
            )
        
        plt.title(f"t-SNE (Layers={embed_layers}, Perp={perplex})")
        plt.xlabel("TSNE Dimension 1")
        plt.ylabel("TSNE Dimension 2")

        handles = []
        legend_labels = []
        for l in unique_labels:
            handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=label_to_color[l], markersize=8))
            legend_labels.append(label_names[l])
        plt.legend(handles, legend_labels, title="Classes", bbox_to_anchor=(1.05, 1), 
                   loc="upper left", ncol=2)
        
        plt.tight_layout()
        out_filename = f"plot_exp_direct_crop_small_letters/{embed_layers}/p{perplex}_i{n_iter}.png"
        plt.savefig(out_filename, dpi=300)
        plt.close()
        

print("All experiments completed.")
