"""
Fetal Ultrasound Dataset Dataloader
Προσαρμοσμένο για Multi-class Segmentation (Background, Brain, CSP, LV)
με ανάγνωση των splits/folds από αρχεία CSV.
"""
import os
import csv
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from augmentation import (
    DoubleCompose,
    DoubleHorizontalFlip,
    DoubleVerticalFlip,
    DoubleRandomRotation,
    DoubleElasticTransform,
    RandomBrightnessContrast,
    GaussianNoise
)


def _seed_worker(worker_id):
    """
    Εξασφαλίζει ότι κάθε παράλληλος worker του DataLoader έχει ντετερμινιστικό seed
    για τα Python `random` και `numpy` κατά το Data Augmentation (πλήρης αναπαραγωγιμότητα).
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class FetalUltrasoundDataset(Dataset):
    def __init__(self, csv_path, data_dir, image_size=(256, 256), augment=False):
        """
        csv_path: Μονοπάτι αρχείου CSV του split/fold (π.χ. ~/Dataset/splits/fold_0_train.csv)
        data_dir: Ριζικός φάκελος της βάσης (π.χ. ~/Dataset)
        image_size: Διαστάσεις εισόδου στο δίκτυο (προεπιλογή: 256x256)
        augment: Εφαρμογή Data Augmentation (True ΜΟΝΟ στο training set)
        """
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.image_size = image_size
        self.augment = augment

        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Δεν βρέθηκε το αρχείο CSV του split: {csv_path}")

        # Ανάγνωση των εγγραφών του συγκεκριμένου split/fold από το CSV
        self.samples = []
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = os.path.join(data_dir, row['image_rel_path'])
                mask_path = os.path.join(data_dir, row['mask_rel_path'])
                self.samples.append({
                    'image_id': row['image_id'],
                    'category': row.get('category', ''),
                    'patient_id': row.get('patient_id', ''),
                    'image_path': img_path,
                    'mask_path': mask_path,
                })

        # Αντιστοίχιση RGB χρωμάτων σε δείκτες κλάσεων
        self.color_to_class = {
            (0, 0, 0): 0,       # Background (Μαύρο)
            (255, 0, 0): 1,     # Brain (Κόκκινο)
            (255, 255, 0): 2,   # CSP (Κίτρινο)
            (0, 255, 0): 2,     # CSP (Πράσινο - για πλήρη συμβατότητα)
            (0, 0, 255): 3      # LV (Μπλε)
        }

        # Ορίζουμε τους μετασχηματισμούς Data Augmentation προσαρμοσμένους σε υπερήχους εμβρυϊκής κεφαλής
        if self.augment:
            self.geom_transforms = DoubleCompose([
                DoubleHorizontalFlip(p=0.5),
                # DoubleVerticalFlip(p=0.5),  # Ανενεργό: θα ανέστρεφε αφύσικα τον κώνο της δέσμης υπερήχων
                DoubleRandomRotation(degrees=15, p=0.5),
                DoubleElasticTransform(alpha_range=(30, 60), sigma_range=(8, 12), p=0.3)
            ])
            self.brightness_contrast = RandomBrightnessContrast(brightness=0.15, contrast=0.15, p=0.5)
            self.noise_transform = GaussianNoise(mean=0., std=0.02, p=0.3)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['image_path']
        mask_path = sample['mask_path']

        # 1. Φόρτωση εικόνας (σε ασπρόμαυρο "L") και μάσκας (σε "RGB" για τα χρώματα)
        image = Image.open(img_path).convert("L")
        mask_rgb = Image.open(mask_path).convert("RGB")

        # 2. Resize στα 256x256
        target_w, target_h = self.image_size
        image = image.resize((target_w, target_h), Image.BILINEAR)
        # Στη μάσκα χρησιμοποιούμε Image.NEAREST για να μην αλλοιωθούν/αναμειχθούν τα χρώματα!
        mask_rgb = mask_rgb.resize((target_w, target_h), Image.NEAREST)

        # 3. Μετατροπή σε Numpy Arrays
        image_np = np.array(image, dtype=np.float32) / 255.0  # Κανονικοποίηση στο [0,1]
        mask_np = np.array(mask_rgb)

        # 4. Γρήγορη vectorized μετατροπή της μάσκας σε πίνακα με ακέραιες κλάσεις (0, 1, 2, 3)
        mask_class = np.zeros((target_h, target_w), dtype=np.int64)
        r = mask_np[:, :, 0] > 128
        g = mask_np[:, :, 1] > 128
        b = mask_np[:, :, 2] > 128

        mask_class[r & ~g & ~b] = 1   # Κλάση 1: Brain (Κόκκινο)
        mask_class[g & ~b] = 2        # Κλάση 2: CSP (Κίτρινο 255,255,0 ή Πράσινο 0,255,0)
        mask_class[b & ~r & ~g] = 3   # Κλάση 3: LV (Μπλε)

        # 5. Μετατροπή σε PyTorch Tensors
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)    # Σχήμα: [1, 256, 256]
        mask_tensor = torch.from_numpy(mask_class).unsqueeze(0)   # Σχήμα: [1, 256, 256] - Απαραιτητο για augmentation

        # 6. Εφαρμογη Data Augmentation (ΜΟΝΟ όταν self.augment == True, δηλ. στο training set)
        if self.augment:
            # Οι γεωμετρικες παραμορφωσεις εφαρμοζονται 100% ταυτοχρονα και στα δυο
            image_tensor, mask_tensor = self.geom_transforms(image_tensor, mask_tensor)
            # Οι φωτομετρικοι μετασχηματισμοι & ο θορυβος εφαρμοζονται αυστηρα μονο στην εικονα του υπερηχου
            image_tensor = self.brightness_contrast(image_tensor)
            image_tensor = self.noise_transform(image_tensor)

        # 7. Αφαιρεση της προσωρινης διαστασης απο τη μασκα(ωστε να μην κρασαρει το loss στο train.py)
        mask_tensor = mask_tensor.squeeze(0).long()

        return image_tensor, mask_tensor


# Η συνάρτηση που καλείται από το train.py
def get_dataloader(
    csv_path,
    data_dir,
    batch_size=4,
    augment=False,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    seed=None
):
    dataset = FetalUltrasoundDataset(
        csv_path=csv_path,
        data_dir=data_dir,
        augment=augment
    )

    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        worker_init_fn = _seed_worker

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,          # Ανακατεύει τις εικόνες μόνο στο training (όχι στο validation/test)
        num_workers=num_workers,  # Παράλληλη φόρτωση batches από τη CPU
        pin_memory=pin_memory,    # Ταχύτερη μεταφορά μνήμης από τη RAM στην GPU (CUDA)
        worker_init_fn=worker_init_fn,
        generator=generator
    )
    return loader