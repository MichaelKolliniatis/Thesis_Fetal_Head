"""
Data Augmentation Transformations for Multi-Class Fetal Ultrasound Segmentation
Ορίζει γεωμετρικούς μετασχηματισμούς που εφαρμόζονται ταυτόχρονα σε εικόνα και μάσκα,
καθώς και φωτομετρικούς μετασχηματισμούς που εφαρμόζονται αποκλειστικά στην εικόνα.
"""
import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from scipy.ndimage import map_coordinates, gaussian_filter


class GaussianNoise:
    """Προσθήκη τυχαίου Γκαουσιανού θορύβου στην εικόνα με περιορισμό τιμών στο [0, 1]."""

    def __init__(self, mean=0., std=0.02, p=0.3):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if random.random() < self.p:
            noise = torch.randn_like(tensor) * self.std + self.mean
            tensor = torch.clamp(tensor + noise, 0.0, 1.0)
        return tensor

    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std}, p={self.p})'


class RandomBrightnessContrast:
    """
    Τυχαία μεταβολή φωτεινότητας (Gain) και αντίθεσης (Dynamic Range)
    για προσομοίωση διαφορετικών ρυθμίσεων υπερηχογράφου.
    """

    def __init__(self, brightness=0.15, contrast=0.15, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, tensor):
        if random.random() < self.p:
            alpha = 1.0 + random.uniform(-self.contrast, self.contrast)
            beta = random.uniform(-self.brightness, self.brightness)
            mean_val = tensor.mean()
            tensor = (tensor - mean_val) * alpha + mean_val + beta
            tensor = torch.clamp(tensor, 0.0, 1.0)
        return tensor

    def __repr__(self):
        return (
            f'{self.__class__.__name__}'
            f'(brightness={self.brightness}, contrast={self.contrast}, p={self.p})'
        )


class DoubleToTensor:
    """Μετατροπή εικόνας και μάσκας σε PyTorch Tensors."""

    def __init__(self):
        pass

    def __call__(self, image, mask, weight=None):
        if weight is None:
            return TF.to_tensor(image), TF.to_tensor(mask)
        weight = weight.view(1, *weight.shape)
        return TF.to_tensor(image), TF.to_tensor(mask), weight

    def __repr__(self):
        return self.__class__.__name__ + '()'


class DoubleHorizontalFlip:
    """Οριζόντια αναστροφή (καθρεφτισμός αριστερά-δεξιά) ταυτόχρονα σε εικόνα και μάσκα."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask, weight=None):
        do_flip = random.random() < self.p
        if do_flip:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            if weight is not None:
                weight = TF.hflip(weight)
        return (image, mask) if weight is None else (image, mask, weight)

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'


class DoubleVerticalFlip:
    """Κατακόρυφη αναστροφή (πάνω-κάτω) ταυτόχρονα σε εικόνα και μάσκα."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask, weight=None):
        do_flip = random.random() < self.p
        if do_flip:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            if weight is not None:
                weight = TF.vflip(weight)
        return (image, mask) if weight is None else (image, mask, weight)

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'


class DoubleRandomRotation:
    """
    Τυχαία περιστροφή κατά [-degrees, +degrees] ταυτόχρονα σε εικόνα και μάσκα.
    Χρησιμοποιεί Bilinear παρεμβολή στην εικόνα και αυστηρά Nearest στη μάσκα
    ώστε να μην αλλοιώνονται οι ακέραιες ετικέτες των κλάσεων (0, 1, 2, 3).
    """

    def __init__(self, degrees=15, p=0.5):
        self.degrees = degrees
        self.p = p

    def __call__(self, image, mask, weight=None):
        if random.random() < self.p:
            angle = random.uniform(-self.degrees, self.degrees)
            image = TF.rotate(
                image, angle, interpolation=InterpolationMode.BILINEAR, fill=0.0
            )
            mask = TF.rotate(
                mask.float(), angle, interpolation=InterpolationMode.NEAREST, fill=0.0
            ).round().long()
            if weight is not None:
                weight = TF.rotate(
                    weight, angle, interpolation=InterpolationMode.BILINEAR, fill=0.0
                )
        return (image, mask) if weight is None else (image, mask, weight)

    def __repr__(self):
        return f'{self.__class__.__name__}(degrees={self.degrees}, p={self.p})'


class DoubleElasticTransform:
    """
    Ήπια ελαστική παραμόρφωση προσαρμοσμένη σε εικόνες 256x256,
    ώστε να εισάγει φυσική ανατομική ποικιλομορφία χωρίς να εξαφανίζει μικρές δομές (CSP).
    """

    def __init__(self, alpha_range=(30, 60), sigma_range=(8, 12), p=0.3):
        self.alpha_range = alpha_range
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, image, mask, weight=None):
        if random.random() < self.p:
            alpha = random.uniform(*self.alpha_range)
            sigma = random.uniform(*self.sigma_range)

            dim = image.shape  # (1, H, W)
            h, w = dim[1], dim[2]

            dx = alpha * gaussian_filter(
                (np.random.rand(h, w) * 2 - 1), sigma, mode="constant", cval=0
            )
            dy = alpha * gaussian_filter(
                (np.random.rand(h, w) * 2 - 1), sigma, mode="constant", cval=0
            )

            rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            indices = np.reshape(rows + dy, (-1, 1)), np.reshape(cols + dx, (-1, 1))

            img_np = image.reshape(h, w).numpy()
            msk_np = mask.reshape(h, w).numpy()

            # order=1 (bilinear) για την εικόνα, order=0 (nearest) για τη μάσκα κλάσεων
            img_warped = map_coordinates(img_np, indices, order=1, mode='constant', cval=0.0).reshape(dim)
            msk_warped = map_coordinates(msk_np, indices, order=0, mode='constant', cval=0).reshape(dim)

            image = torch.from_numpy(img_warped.astype(np.float32))
            image = torch.clamp(image, 0.0, 1.0)
            mask = torch.from_numpy(msk_warped.astype(np.int64))

            if weight is not None:
                w_np = weight.reshape(h, w).numpy()
                w_warped = map_coordinates(w_np, indices, order=1, mode='constant', cval=0.0).reshape(dim)
                weight = torch.from_numpy(w_warped.astype(np.float32))

        return (image, mask) if weight is None else (image, mask, weight)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}'
            f'(alpha_range={self.alpha_range}, sigma_range={self.sigma_range}, p={self.p})'
        )


class DoubleCompose(transforms.Compose):

    def __call__(self, image, mask, weight=None):
        if weight is None:
            for t in self.transforms:
                image, mask = t(image, mask)
            return image, mask
        for t in self.transforms:
            image, mask, weight = t(image, mask, weight)
        return image, mask, weight
