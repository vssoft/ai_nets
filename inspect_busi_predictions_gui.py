import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import torch

from unet_busi_loader import load_busi_dataset
from unet_model import UNet


# -------------------------
# CONFIG
# -------------------------
DATA_PATH = "D:/pyton/datasets/Dataset_BUSI_with_GT"
FOLDER_PATHS = [f"{DATA_PATH}/malignant", f"{DATA_PATH}/benign"]
IMG_SIZE = 128
THRESHOLD = 0.5
CHECKPOINT_PATH = "checkpoints_busi/unet_busi_best.pt"

OVERLAY_ALPHA = 0.35


# -------------------------
# DEVICE
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------------------------
# LOAD DATA + MODEL
# -------------------------
X, y = load_busi_dataset(FOLDER_PATHS, size=IMG_SIZE, verbose=False)
print("Loaded dataset:", X.shape, y.shape)

model = UNet(in_channels=1, out_channels=1, base_filters=64).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()
print("Loaded model:", CHECKPOINT_PATH)


@torch.no_grad()
def predict_prob(idx):
    """Return probability map [H,W] for one sample index."""
    img = torch.tensor(X[idx:idx+1], dtype=torch.float32).to(device)
    logits = model(img)
    prob = torch.sigmoid(logits).cpu().numpy()[0, 0]  # [H,W]
    return prob


def dice_iou(pred_bin, gt_bin, eps=1e-7):
    """Compute Dice and IoU for one image (binary masks)."""
    pred = pred_bin.astype(np.uint8)
    gt = gt_bin.astype(np.uint8)

    inter = (pred & gt).sum()
    dice = (2 * inter + eps) / (pred.sum() + gt.sum() + eps)

    union = pred.sum() + gt.sum() - inter
    iou = (inter + eps) / (union + eps)

    return float(dice), float(iou)


class Viewer:
    def __init__(self):
        self.idx = 1
        self.max_idx = len(X) - 1

        # Create 4 panels:
        # Input | Ground Truth | Predicted Mask | Overlay
        self.fig, self.ax = plt.subplots(1, 4, figsize=(14, 4))

        # Reserve more space at the bottom for controls + info text
        plt.subplots_adjust(bottom=0.40)

        # ---- Buttons ----
        left = 0.16
        step = 0.10
        # [left, bottom, width, height]
        axprev = plt.axes([left + step * 0, 0.16, 0.09, 0.08])
        axnext = plt.axes([left + step * 1, 0.16, 0.09, 0.08])
        axrand = plt.axes([left + step * 2, 0.16, 0.09, 0.08])
        axexit = plt.axes([left + step * 3, 0.16, 0.09, 0.08])

        self.bprev = Button(axprev, "Previous")
        self.bnext = Button(axnext, "Next")
        self.brand = Button(axrand, "Random")
        self.bexit = Button(axexit, "Exit")

        # Connect button callbacks
        self.bprev.on_clicked(self.prev)
        self.bnext.on_clicked(self.next)
        self.brand.on_clicked(self.random)
        self.bexit.on_clicked(self.exit)

        # ---- Jump-to-index TextBox (align with buttons) ----
        axtxt = plt.axes([0.60, 0.17, 0.12, 0.06])
        self.textbox = TextBox(axtxt, "Index", initial=str(self.idx))
        self.textbox.on_submit(self.jump_to_index)

        # ---- Info text ----
        # self.info_text = self.fig.text(0.02, 0.02, "", fontsize=10, family="monospace")
        self.info_text = self.fig.text(0.02, 0.03, "", fontsize=10, family="monospace")
        # ---- Keyboard shortcuts ----
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        # First draw
        self.update()

    def update(self):
        """Update display for current index."""
        idx = self.idx

        x_img = X[idx, 0]
        gt = y[idx, 0]

        prob = predict_prob(idx)
        pred_bin = (prob > THRESHOLD).astype(np.uint8)
        gt_bin = (gt > 0.5).astype(np.uint8)

        dice, iou = dice_iou(pred_bin, gt_bin)

        # Clear all axes and enable coordinate axes
        for a in self.ax:
            a.clear()
            a.set_xlabel("x (pixels)")
            a.set_ylabel("y (pixels)")
            # a.grid(True)  # set True if you want grid lines
            # a.grid(True, linewidth=0.3)
            a.set_xticks(range(0, IMG_SIZE, 20))
            a.set_yticks(range(0, IMG_SIZE, 20))

        # Input
        self.ax[0].imshow(x_img, cmap="gray")
        self.ax[0].set_title(f"Input (idx={idx})")
        self.ax[0].set_xlim(0, IMG_SIZE - 1)
        self.ax[0].set_ylim(IMG_SIZE - 1, 0)

        # Ground truth
        self.ax[1].imshow(gt, cmap="gray")
        self.ax[1].set_title("Ground Truth")
        self.ax[1].set_xlim(0, IMG_SIZE - 1)
        self.ax[1].set_ylim(IMG_SIZE - 1, 0)

        # Prediction
        self.ax[2].imshow(pred_bin, cmap="gray")
        self.ax[2].set_title(f"Prediction (thr={THRESHOLD})")
        self.ax[2].set_xlim(0, IMG_SIZE - 1)
        self.ax[2].set_ylim(IMG_SIZE - 1, 0)

        # Overlay
        self.ax[3].imshow(x_img, cmap="gray")
        self.ax[3].imshow(pred_bin, alpha=OVERLAY_ALPHA)
        self.ax[3].set_title(f"Overlay (alpha={OVERLAY_ALPHA})")
        self.ax[3].set_xlim(0, IMG_SIZE - 1)
        self.ax[3].set_ylim(IMG_SIZE - 1, 0)

        # Update textbox without triggering events (prevents Windows freeze)
        self.textbox.eventson = False
        self.textbox.set_val(str(idx))
        self.textbox.eventson = True

        # Update info text
        self.info_text.set_text(
            f"Index: {idx}/{self.max_idx} | Dice: {dice:.4f} | IoU: {iou:.4f}\n"
            "Keys: Right/D=Next | Left/A=Prev | R=Random | Enter in textbox=Jump"
        )

        # Redraw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def next(self, event=None):
        """Next sample."""
        self.idx += 1
        if self.idx > self.max_idx:
            self.idx = 0
        self.update()

    def prev(self, event=None):
        """Previous sample."""
        self.idx -= 1
        if self.idx < 0:
            self.idx = self.max_idx
        self.update()

    def random(self, event=None):
        """Random sample."""
        self.idx = np.random.randint(0, self.max_idx + 1)
        self.update()

    def jump_to_index(self, text):
        """Jump to index from TextBox input."""
        try:
            idx = int(text)
            if 0 <= idx <= self.max_idx:
                self.idx = idx
                self.update()
            else:
                print(f"Index out of range. Must be 0..{self.max_idx}")
        except ValueError:
            print("Invalid index. Please enter an integer.")

    def on_key(self, event):
        """Keyboard shortcut handler."""
        if event.key in ["right", "d"]:
            self.next()
        elif event.key in ["left", "a"]:
            self.prev()
        elif event.key == "r":
            self.random()

    def exit(self, event=None):
        """Close the GUI window and exit the program."""
        plt.close(self.fig)

if __name__ == "__main__":
    Viewer()
    plt.show()
