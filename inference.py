"""
Model Inference & Qualitative Visualization
Φορτώνει ένα εκπαιδευμένο checkpoint του UNet (4 κλάσεων) και παράγει
συγκριτικές εικόνες (Υπέρηχος | Ground Truth | Πρόβλεψη | Υπέρθεση/Overlay)
αποθηκεύοντάς τες στον φάκελο `visualization/` (headless-safe για SSH/SLURM server).
"""
import os
import csv
import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw

from unet import UNet


# Χρωματική παλέτα RGB για οπτικοποίηση των κλάσεων (0..3)
PALETTE_RGB = np.array([
    [0,   0,   0],     # 0: Background (Μαύρο)
    [255, 0,   0],     # 1: Brain      (Κόκκινο)
    [255, 255, 0],     # 2: CSP        (Κίτρινο)
    [0,   0,   255],   # 3: LV         (Μπλε)
], dtype=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Make segmentation predictions and save visualizations'
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help='όνομα αρχείου checkpoint στον φάκελο models/ (προεπιλογή: UNet_final_dev80.pt ή UNet_fold0_best.pt)'
    )
    parser.add_argument(
        '--data-dir', type=str, default=os.path.expanduser('~/Dataset'),
        help='ριζικός φάκελος της βάσης (default: ~/Dataset)'
    )
    parser.add_argument(
        '--csv', type=str, default=None,
        help='αρχείο CSV από όπου θα επιλεγούν εικόνες (default: <data-dir>/splits/test.csv)'
    )
    parser.add_argument(
        '--image-id', type=str, default=None,
        help='προαιρετικά: όνομα συγκεκριμένης εικόνας για οπτικοποίηση (π.χ. Patient00188_Plane3_2_of_3.png)'
    )
    parser.add_argument(
        '--num-samples', type=int, default=8,
        help='πόσες ενδεικτικές εικόνες να αποθηκευτούν στον φάκελο visualization/ (default: 8)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='visualization',
        help='φάκελος αποθήκευσης των εικόνων αποτελεσμάτων (default: visualization/)'
    )
    parser.add_argument(
        '--no-cuda', action='store_true', default=False,
        help='εκτέλεση σε CPU'
    )
    return parser.parse_args()


def rgb_mask_to_class_indices(mask_np):
    """Μετατρέπει RGB μάσκα (H, W, 3) σε ακέραιους δείκτες κλάσεων (0, 1, 2, 3)."""
    h, w, _ = mask_np.shape
    mask_class = np.zeros((h, w), dtype=np.int64)
    r = mask_np[:, :, 0] > 128
    g = mask_np[:, :, 1] > 128
    b = mask_np[:, :, 2] > 128

    mask_class[r & ~g & ~b] = 1   # Brain
    mask_class[g & ~b] = 2        # CSP
    mask_class[b & ~r & ~g] = 3   # LV
    return mask_class


def class_indices_to_rgb(mask_class):
    """Μετατρέπει πίνακα κλάσεων (H, W) με τιμές 0..3 σε έγχρωμη εικόνα RGB (H, W, 3)."""
    return PALETTE_RGB[np.clip(mask_class, 0, len(PALETTE_RGB) - 1)]


def compute_sample_dice(pred_class, gt_class, n_classes=4):
    """Υπολογίζει το μέσο Dice Score (στις δομές που υπάρχουν στην εικόνα) για εμφάνιση στον τίτλο."""
    dices = []
    for c in range(1, n_classes):
        p = (pred_class == c)
        g = (gt_class == c)
        if not p.any() and not g.any():
            continue
        inter = np.logical_and(p, g).sum()
        denom = p.sum() + g.sum()
        dices.append((2.0 * inter) / max(denom, 1))
    return float(np.mean(dices)) if dices else 1.0


def predict(image_np, model, device):
    """
    Εκτελεί πρόβλεψη σε κανονικοποιημένη εικόνα 256x256 (εύρος [0, 1]),
    ακριβώς με το ίδιο preprocessing του dataloader.py.
    """
    model.eval()
    tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 256, 256)
    with torch.no_grad():
        logits = model(tensor)
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy()
    return pred


def save_visualization(image_np, gt_class, pred_class, sample_info, out_path):
    """
    Αποθηκεύει συγκριτική εικόνα 4 στηλών (καθαρά με PIL/Pillow, χωρίς εξάρτηση από matplotlib):
      1. Εικόνα Υπερήχου (Ultrasound 256x256)
      2. Πραγματική Μάσκα (Ground Truth)
      3. Πρόβλεψη Μοντέλου (Prediction)
      4. Υπέρθεση Πρόβλεψης πάνω στον Υπέρηχο (Prediction Overlay)
    """
    gt_rgb = class_indices_to_rgb(gt_class)
    pred_rgb = class_indices_to_rgb(pred_class)

    # Δημιουργία RGB εικόνας υπερήχου και Overlay
    us_rgb = np.stack([np.clip(image_np * 255.0, 0, 255)] * 3, axis=-1).astype(np.uint8)
    fg_mask = pred_class > 0
    overlay_rgb = us_rgb.astype(np.float32).copy()
    overlay_rgb[fg_mask] = 0.55 * overlay_rgb[fg_mask] + 0.45 * pred_rgb[fg_mask].astype(np.float32)
    overlay_rgb = np.clip(overlay_rgb, 0, 255).astype(np.uint8)

    sample_dice = compute_sample_dice(pred_class, gt_class)

    # Διαστάσεις καμβά (4 πάνελ των 256x256 με περιθώρια, τίτλους πάνω και υπόμνημα κάτω)
    panel_w, panel_h = 256, 256
    pad = 16
    top_bar = 58
    bottom_bar = 40
    canvas_w = 4 * panel_w + 5 * pad
    canvas_h = top_bar + panel_h + bottom_bar

    canvas = Image.new('RGB', (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Κεντρικός τίτλος
    header_text = (
        f"Image: {sample_info['image_id']}   |   "
        f"Category: {sample_info['category']}   |   "
        f"Mean Dice: {sample_dice:.4f}"
    )
    draw.text((pad, 10), header_text, fill=(0, 0, 0))

    panels = [
        ("1. Ultrasound (256x256)", us_rgb),
        ("2. Ground Truth", gt_rgb),
        ("3. Model Prediction", pred_rgb),
        ("4. Prediction Overlay", overlay_rgb),
    ]

    for i, (title, arr) in enumerate(panels):
        x0 = pad + i * (panel_w + pad)
        y0 = top_bar
        draw.text((x0 + 4, y0 - 20), title, fill=(30, 30, 30))
        panel_img = Image.fromarray(arr, mode='RGB')
        canvas.paste(panel_img, (x0, y0))
        draw.rectangle([x0 - 1, y0 - 1, x0 + panel_w, y0 + panel_h], outline=(180, 180, 180))

    # Υπόμνημα χρωμάτων (Legend) στο κάτω μέρος
    legend_y = top_bar + panel_h + 12
    legend_items = [
        ((255, 0, 0), "Brain (Red)"),
        ((255, 255, 0), "CSP (Yellow)"),
        ((0, 0, 255), "LV (Blue)"),
    ]
    lx = pad
    for color, label in legend_items:
        draw.rectangle([lx, legend_y, lx + 16, legend_y + 14], fill=color, outline=(0, 0, 0))
        draw.text((lx + 22, legend_y + 1), label, fill=(0, 0, 0))
        lx += 160

    canvas.save(out_path)


def resolve_checkpoint_path(model_arg):
    """Βρίσκει το κατάλληλο checkpoint στον φάκελο models/."""
    models_dir = os.path.join(os.getcwd(), 'models')
    if model_arg is not None:
        candidate = model_arg if os.path.isfile(model_arg) else os.path.join(models_dir, model_arg)
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError(f"Δεν βρέθηκε το checkpoint: {candidate}")

    # Αυτόματη αναζήτηση αν δεν δόθηκε --model
    for default_name in ('UNet_final_dev80.pt', 'UNet_fold0_best.pt'):
        candidate = os.path.join(models_dir, default_name)
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"Δεν βρέθηκε εκπαιδευμένο μοντέλο στον φάκελο {models_dir}. "
        f"Δώσε το όνομα με `--model <checkpoint.pt>`."
    )


def select_samples(rows, image_id=None, num_samples=8):
    """
    Επιλέγει είτε μία συγκεκριμένη εικόνα (--image-id),
    είτε ισόρροπα δείγματα από όλες τις 4 κατηγορίες της βάσης.
    """
    if image_id is not None:
        matched = [r for r in rows if r['image_id'] == image_id]
        if not matched:
            raise ValueError(f"Η εικόνα {image_id} δεν βρέθηκε μέσα στο επιλεγμένο CSV!")
        return matched

    # Ισόρροπη επιλογή δειγμάτων από τις 4 κατηγορίες
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r.get('category', 'Unknown'), []).append(r)

    selected = []
    cats = sorted(by_cat.keys())
    idx_per_cat = {c: 0 for c in cats}

    while len(selected) < min(num_samples, len(rows)):
        added = False
        for c in cats:
            if idx_per_cat[c] < len(by_cat[c]) and len(selected) < num_samples:
                selected.append(by_cat[c][idx_per_cat[c]])
                idx_per_cat[c] += 1
                added = True
        if not added:
            break

    return selected


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    ckpt_path = resolve_checkpoint_path(args.model)
    csv_path = args.csv if args.csv is not None else os.path.join(args.data_dir, 'splits', 'test.csv')

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Δεν βρέθηκε το αρχείο CSV: {csv_path}")

    print(f"Φόρτωση μοντέλου από: {ckpt_path} (Device: {device})")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    n_classes = checkpoint.get('n_classes', 4)

    model = UNet(n_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with open(csv_path, mode='r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    samples = select_samples(rows, image_id=args.image_id, num_samples=args.num_samples)
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_tag = os.path.splitext(os.path.basename(ckpt_path))[0]

    print(f"Παραγωγή {len(samples)} οπτικοποιήσεων από το {csv_path}...")
    for sample in samples:
        img_path = os.path.join(args.data_dir, sample['image_rel_path'])
        mask_path = os.path.join(args.data_dir, sample['mask_rel_path'])

        image = Image.open(img_path).convert("L").resize((256, 256), Image.BILINEAR)
        mask_rgb = Image.open(mask_path).convert("RGB").resize((256, 256), Image.NEAREST)

        image_np = np.array(image, dtype=np.float32) / 255.0
        gt_class = rgb_mask_to_class_indices(np.array(mask_rgb))
        pred_class = predict(image_np, model, device)

        base_name = os.path.splitext(sample['image_id'])[0]
        out_file = os.path.join(args.output_dir, f"{ckpt_tag}_{base_name}.png")
        save_visualization(image_np, gt_class, pred_class, sample, out_file)
        print(f"  -> Αποθηκεύτηκε: {out_file}")

    print(f"\nΟλοκληρώθηκε! Οι εικόνες βρίσκονται στον φάκελο: {os.path.abspath(args.output_dir)}/")
