import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
from pathlib import Path
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
from dataset import xywhn_to_xyxy

def plot_and_save_cropped_feature_map(cropped_feats, plot_image_name):

    B, C, _, _ = cropped_feats.shape
    if B != 1:
        print("Warning: This function expects a single batch (B=1). Taking the first element.")

    fm = cropped_feats[0]

    n = C  
    rows = math.ceil(n / 8)
    cols = min(n, 8)

    fig, ax = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
    ax = ax.ravel() if n > 1 else [ax]  

    for i in range(n):
        ax[i].axis('off')
        ax[i].imshow(fm[i].cpu().numpy(), cmap='gray')
        ax[i].set_title(f"Channel {i}")

    for j in range(n, len(ax)):
        ax[j].axis('off')

    plt.tight_layout()
    fig.savefig(plot_image_name, dpi=fig.dpi)
    plt.close(fig)


def feature_visualization(x, bbox, stage, n=4, save_dir=Path("feature_plots_1_6_15_21")):
     print("stage",stage)

     if isinstance(x, torch.Tensor):
        
        _, channels, height, width = x.shape  # batch, channels, height, width
        if height > 1 and width > 1:
            f = f"{save_dir}/{stage}_features.png"

            # Chunk the tensor into individual channel blocks
            blocks = torch.chunk(x[0].cpu(), channels, dim=0)
            n = min(n, channels)  # number of channels to plot

            fig, ax = plt.subplots(math.ceil(n / 4), 4, tight_layout=True)
            ax = ax.ravel()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            
            for i in range(n):
                print("blocks[i]",blocks[i].shape)
                ax[i].imshow(blocks[i].squeeze())

                if bbox is not None:
                    for each_bbox in bbox:
                        xmin, ymin, xmax, ymax = xywhn_to_xyxy(each_bbox, height, width)
                        rect_width = xmax - xmin
                        rect_height = ymax - ymin
                        rect = patches.Rectangle((xmin, ymin), rect_width, rect_height,
                                                linewidth=1, edgecolor='r', facecolor='none')
                        ax[i].add_patch(rect)
                
                ax[i].axis("off")

            plt.savefig(f, dpi=300, bbox_inches="tight")
            plt.close()


def plot_confusion_matrix(all_labels, all_preds):

    class_labels = sorted(set(all_labels) | set(all_preds))
    
    cm = confusion_matrix(all_labels, all_preds, labels=class_labels)
    print("Confusion Matrix:\n", cm)

    sns.set_theme(rc={'figure.figsize': (20, 20)})
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=class_labels, yticklabels=class_labels)

    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")


    # plt.figure(figsize=(20, 20))
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    # disp.plot(cmap=plt.cm.Blues)
    
    plt.title("Confusion Matrix")

    plt.savefig('results/test_confusion_matrix_feature_crop.png')


def plot_training_loss(train_losses,model_type, results_folder="results"):

    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.title(f"Epoch vs {model_type}Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel(f"{model_type} Training Loss")
    plt.legend()

    plt.savefig(f'{results_folder}/training_Loss_{model_type}.png')