"""
Model Training & Evaluation Pipeline
Αυτό το script ορίζει τη διαδικασία εκπαίδευσης και αξιολόγησης του μοντέλου.
Προσαρμόστηκε για το Fetal Head Ultrasound Dataset με:
  1. Σταθερά seeds (Reproducibility: Python, NumPy, PyTorch, CUDA)
  2. 5-Fold Cross-Validation (`--mode cv`) με πλήρη επαναρχικοποίηση ανά fold,
     Early Stopping στο validation set και εξαγωγή αποτελεσμάτων ως mean ± std
  3. Τελική εκπαίδευση σε ολόκληρο το 80% Development Set (`--mode final_train`)
  4. Εφάπαξ τελική αξιολόγηση στο ανεξάρτητο 20% Test Set (`--mode test`)
"""
import os
import copy
import random
import argparse
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from unet import UNet
from dataloader import get_dataloader
from metric import iou, dice_coef, hausdorff_distance, average_surface_distance, compute_model_complexity
from loss import CEDiceLoss


CLASS_NAMES = ['Background', 'Brain', 'CSP', 'LV']


def seed_everything(seed=42):
    """
    Ορίζει σταθερά seeds σε Python, NumPy, PyTorch και CUDA
    για πλήρη αναπαραγωγιμότητα (Reproducibility) των πειραμάτων.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train & Evaluate Fetal Ultrasound Segmentation')
    parser.add_argument(
        '--mode', type=str, default='cv', choices=['cv', 'final_train', 'test'],
        help='mode εκτέλεσης: "cv" (5-fold CV στο 80%), "final_train" (εκπαίδευση σε όλο το 80%), "test" (αξιολόγηση στο 20% test set)'
    )
    parser.add_argument(
        '--fold', type=int, default=None, choices=[0, 1, 2, 3, 4],
        help='εκτέλεση συγκεκριμένου fold (0..4) στο mode="cv". Αν παραλειφθεί, τρέχουν σειριακά όλα τα 5 folds (0..4)'
    )
    parser.add_argument(
        '--skip-completed', action='store_true', default=False,
        help='προσπερνάει όσα folds έχουν ήδη αποθηκευμένο checkpoint στον φάκελο models/'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='βασικό random seed αναπαραγωγιμότητας (προεπιλογή: 42 -> στα 5 folds χρησιμοποιούνται τα seeds 42..46)'
    )
    parser.add_argument(
        '--patience', type=int, default=15,
        help='αριθμός epochs υπομονής για Early Stopping βάσει Validation Dice (0 = απενεργοποίηση, προεπιλογή: 15)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=4, metavar='N',
        help='input batch size for training (default: 4)'
    )
    parser.add_argument(
        '--test-batch-size', type=int, default=4, metavar='N',
        help='input batch size for validation/testing (default: 4)'
    )
    parser.add_argument(
        '--epochs', type=int, default=50, metavar='N',
        help='number of epochs to train (default: 50)'
    )
    parser.add_argument(
        '--lr', type=float, default=0.0001, metavar='LR',
        help='learning rate (default: 0.0001)'
    )
    parser.add_argument(
        '--n-classes', type=int, default=4,
        help='number of segmentation classes (default: 4)'
    )
    parser.add_argument(
        '--num-workers', type=int, default=2,
        help='number of parallel DataLoader workers (default: 2)'
    )
    parser.add_argument(
        '--log-interval', type=int, default=100,
        help='κάθε πόσα batches θα τυπώνεται το training loss (default: 100)'
    )
    parser.add_argument(
        '--no-cuda', action='store_true', default=False,
        help='disables CUDA training'
    )
    parser.add_argument(
        '--save', action='store_true', default=True,
        help='save model checkpoints in models/ directory (default: True)'
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help='όνομα αρχείου checkpoint στον φάκελο models/ (π.χ. για --mode test ή συνέχιση εκπαίδευσης)'
    )
    parser.add_argument(
        '--tensorboard', action='store_true', default=False,
        help='record training log to Tensorboard'
    )
    parser.add_argument(
        '--data-dir', type=str, default=os.path.expanduser('~/Dataset'),
        help='μονοπάτι ριζικού φακέλου της βάσης (default: ~/Dataset)'
    )
    parser.add_argument(
        '--splits-dir', type=str, default=None,
        help='μονοπάτι φακέλου με τα CSV splits (default: <data-dir>/splits)'
    )
    args = parser.parse_args()
    if args.splits_dir is None:
        args.splits_dir = os.path.join(args.data_dir, 'splits')
    return args


def train(model, device, data_loader, optimizer, criterion, epoch, log_interval=100):
    """train model for one epoch"""
    model.train()
    loss_sum = 0.
    num_batches = len(data_loader)

    for step, (images, masks) in enumerate(data_loader):
        # Μεταφορά δεδομένων στην GPU (αν υπάρχει)
        X = images.to(device, non_blocking=True)
        # Η μάσκα πρέπει να είναι ακέραιος (long) για το CrossEntropyLoss
        y = masks.to(device, dtype=torch.long, non_blocking=True)

        # Forward pass
        y_pred = model(X)

        # Back propagation
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

        # Log ανά log_interval batches και στο τελευταίο batch του epoch
        if (step + 1) % log_interval == 0 or (step + 1) == num_batches:
            processed = min((step + 1) * data_loader.batch_size, len(data_loader.dataset))
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, processed, len(data_loader.dataset),
                    100. * (step + 1) / num_batches, loss.item()
                )
            )

    return loss_sum / max(num_batches, 1)


def validate(model, device, data_loader, criterion, n_classes, split_name="Validation"):
    """Evaluate model performance with validation or test data"""
    model.eval()
    test_loss = 0

    total_iou_inter = np.zeros(n_classes)
    total_iou_union = np.zeros(n_classes)
    total_dice_inter = np.zeros(n_classes)
    total_dice_union = np.zeros(n_classes)
    total_hd_sum = np.zeros(n_classes)
    total_hd_count = np.zeros(n_classes)
    total_asd_sum = np.zeros(n_classes)
    total_asd_count = np.zeros(n_classes)

    with torch.no_grad():
        for images, masks in data_loader:
            X = images.to(device, non_blocking=True)
            y = masks.to(device, dtype=torch.long, non_blocking=True)

            y_pred = model(X)
            test_loss += criterion(y_pred, y).item()

            # Βρίσκουμε την κλάση με την υψηλότερη πιθανότητα για κάθε pixel
            pred = torch.argmax(y_pred, dim=1)                  # (B, H, W)
            batch_size = X.shape[0]

            pred = pred.view(batch_size, -1)
            y = y.view(batch_size, -1)

            iou_inter, iou_union = iou(pred, y, n_classes)
            total_iou_inter += iou_inter
            total_iou_union += iou_union

            dice_inter, dice_union = dice_coef(pred, y, n_classes)
            total_dice_inter += dice_inter
            total_dice_union += dice_union

            hd_sum, hd_count = hausdorff_distance(pred, y, n_classes)
            total_hd_sum += hd_sum
            total_hd_count += hd_count

            asd_sum, asd_count = average_surface_distance(pred, y, n_classes)
            total_asd_sum += asd_sum
            total_asd_count += asd_count

    eps = 1e-6
    class_iou = total_iou_inter / (total_iou_union + eps)
    class_dice = (2.0 * total_dice_inter) / (total_dice_union + eps)

    class_hd = np.zeros(n_classes)
    valid_hd_classes = []
    class_asd = np.zeros(n_classes)
    valid_asd_classes = []
    for c in range(1, n_classes):
        if total_hd_count[c] > 0:
            class_hd[c] = total_hd_sum[c] / total_hd_count[c]
            valid_hd_classes.append(class_hd[c])
        if total_asd_count[c] > 0:
            class_asd[c] = total_asd_sum[c] / total_asd_count[c]
            valid_asd_classes.append(class_asd[c])

    test_loss /= max(len(data_loader), 1)

    # Μεσος ορος μονο για τα οργανα, αγνοωντας την κλαση 0 (φοντο)
    avg_iou = np.mean(class_iou[1:])
    avg_dice = np.mean(class_dice[1:])
    avg_hd = np.mean(valid_hd_classes) if len(valid_hd_classes) > 0 else 0.0
    avg_asd = np.mean(valid_asd_classes) if len(valid_asd_classes) > 0 else 0.0

    per_class_iou_str = ' | '.join(
        '{}: {:.4f}'.format(CLASS_NAMES[c] if c < len(CLASS_NAMES) else f'Class {c}', class_iou[c])
        for c in range(1, n_classes)
    )
    per_class_dice_str = ' | '.join(
        '{}: {:.4f}'.format(CLASS_NAMES[c] if c < len(CLASS_NAMES) else f'Class {c}', class_dice[c])
        for c in range(1, n_classes)
    )
    per_class_hd_str = ' | '.join(
        '{}: {:.2f}px'.format(CLASS_NAMES[c] if c < len(CLASS_NAMES) else f'Class {c}', class_hd[c])
        for c in range(1, n_classes)
    )
    per_class_asd_str = ' | '.join(
        '{}: {:.2f}px'.format(CLASS_NAMES[c] if c < len(CLASS_NAMES) else f'Class {c}', class_asd[c])
        for c in range(1, n_classes)
    )

    print(
        f'\n{split_name} set: Average loss: {test_loss:.4f}, '
        + f'Average IOU: {avg_iou:.4f}, '
        + f'Average Dice: {avg_dice:.4f}, '
        + f'Average HD: {avg_hd:.2f}px, '
        + f'Average ASD: {avg_asd:.2f}px\n'
        + f'  IoU ανά δομή  -> {per_class_iou_str}\n'
        + f'  Dice ανά δομή -> {per_class_dice_str}\n'
        + f'  HD ανά δομή   -> {per_class_hd_str}\n'
        + f'  ASD ανά δομή  -> {per_class_asd_str}\n'
    )

    return test_loss, avg_iou, avg_dice, avg_hd, avg_asd, class_iou, class_dice, class_hd, class_asd


def initialize_model(args):
    """Initialize model checkpoint dictionary for storing training progress"""
    model_dict = {
        'total_epoch': args.epochs,
        'n_classes': args.n_classes,
        'model_state_dict': None,
        'optimizer_state_dict': None,
        'train_loss': list(),
        'test_loss': list(),
        'complexity': {
            'total_params': 0,
            'trainable_params': 0,
            'model_size_mb': 0.0,
            'gflops': None
        },
        'metrics': {
            'IOU': list(),
            'class_IOU': list(),
            'Dice': list(),
            'class_Dice': list(),
            'HD': list(),
            'class_HD': list(),
            'ASD': list(),
            'class_ASD': list(),
            'best': {
                'IOU': 0.,
                'class_IOU': None,
                'Dice': 0.,
                'class_Dice': None,
                'HD': 0.,
                'class_HD': None,
                'ASD': 0.,
                'class_ASD': None,
                'epoch': 0
            }
        }
    }
    return model_dict


def get_model(args, device, load_checkpoint=None, print_complexity=True):
    """Initialize or load model checkpoint and initialize model and optimizer from scratch"""
    if load_checkpoint:
        model_path = os.path.join(os.getcwd(), 'models', load_checkpoint)
        model_dict = torch.load(model_path, map_location=device, weights_only=False)
    else:
        model_dict = initialize_model(args)

    n_classes = model_dict['n_classes']
    model = UNet(n_classes).to(device)
    optimizer = optim.Adam(model.parameters(), args.lr)

    if load_checkpoint:
        model.load_state_dict(model_dict['model_state_dict'])
        if model_dict.get('optimizer_state_dict') is not None:
            optimizer.load_state_dict(model_dict['optimizer_state_dict'])

    # Υπολογισμός χωρητικότητας & υπολογιστικής πολυπλοκότητας
    complexity = compute_model_complexity(model, device, input_size=(1, 1, 256, 256))
    model_dict['complexity'] = complexity
    if print_complexity:
        gflops_str = f"{complexity['gflops']:.2f} GFLOPs" if complexity['gflops'] is not None else "N/A"
        print(
            "\n" + "=" * 60 + "\n"
            f"Χωρητικότητα & Πολυπλοκότητα Μοντέλου ({model.name})\n"
            f"  Συνολικές Παράμετροι (Total):      {complexity['total_params']:,} ({complexity['total_params'] / 1e6:.2f}M)\n"
            f"  Εκπαιδεύσιμες Παράμετροι:          {complexity['trainable_params']:,} ({complexity['trainable_params'] / 1e6:.2f}M)\n"
            f"  Μέγεθος Μοντέλου στη Μνήμη:        {complexity['model_size_mb']:.2f} MB\n"
            f"  Υπολογιστική Πολυπλοκότητα (256²): {gflops_str}\n"
            + "=" * 60 + "\n"
        )

    return model, optimizer, model_dict


def print_best_summary(model_name, best_metrics, n_classes, complexity=None, header="ΑΠΟΤΕΛΕΣΜΑΤΑ"):
    """Τυπώνει συγκεντρωτικά τις μετρικές του καλύτερου epoch."""
    print("\n" + "-" * 70)
    print(f"{header} ({model_name})")
    print("-" * 70)
    if complexity:
        gflops_end = f"{complexity['gflops']:.2f} GFLOPs" if complexity.get('gflops') is not None else "N/A"
        print(
            f"Model Complexity ({model_name}) -> "
            f"Params: {complexity['total_params'] / 1e6:.2f}M (Trainable: {complexity['trainable_params'] / 1e6:.2f}M) | "
            f"Size: {complexity['model_size_mb']:.2f} MB | "
            f"Complexity: {gflops_end}"
        )

    best_epoch = best_metrics['epoch']
    best_class_dice = best_metrics.get('class_Dice')
    best_class_iou = best_metrics.get('class_IOU')
    best_class_hd = best_metrics.get('class_HD')
    best_class_asd = best_metrics.get('class_ASD')

    print(f"Best Average Dice Score (Epoch {best_epoch}): {best_metrics['Dice']:.4f}")
    if best_class_dice is not None:
        s = ' | '.join(
            '{}: {:.4f}'.format(CLASS_NAMES[c] if c < len(CLASS_NAMES) else f'Class {c}', best_class_dice[c])
            for c in range(1, n_classes)
        )
        print(f"  Best Dice ανά δομή -> {s}")

    print(f"IoU (at Best Dice): {best_metrics['IOU']:.4f}")
    if best_class_iou is not None:
        s = ' | '.join(
            '{}: {:.4f}'.format(CLASS_NAMES[c] if c < len(CLASS_NAMES) else f'Class {c}', best_class_iou[c])
            for c in range(1, n_classes)
        )
        print(f"  IoU ανά δομή       -> {s}")

    print(f"Hausdorff Distance (at Best Dice): {best_metrics['HD']:.2f}px")
    if best_class_hd is not None:
        s = ' | '.join(
            '{}: {:.2f}px'.format(CLASS_NAMES[c] if c < len(CLASS_NAMES) else f'Class {c}', best_class_hd[c])
            for c in range(1, n_classes)
        )
        print(f"  HD ανά δομή        -> {s}")

    print(f"Average Surface Distance (at Best Dice): {best_metrics['ASD']:.2f}px")
    if best_class_asd is not None:
        s = ' | '.join(
            '{}: {:.2f}px'.format(CLASS_NAMES[c] if c < len(CLASS_NAMES) else f'Class {c}', best_class_asd[c])
            for c in range(1, n_classes)
        )
        print(f"  ASD ανά δομή       -> {s}")
    print("-" * 70 + "\n")


def print_cv_summary_across_folds(fold_best_list, n_classes):
    """
    Υπολογίζει και τυπώνει το mean ± standard deviation (mean ± std)
    όλων των μετρικών πάνω στα 5 folds του Cross-Validation,
    καθώς και τον προτεινόμενο αριθμό epochs για την τελική εκπαίδευση στο 80%.
    """
    n_folds = len(fold_best_list)
    dices = np.array([b['Dice'] for b in fold_best_list])
    ious = np.array([b['IOU'] for b in fold_best_list])
    hds = np.array([b['HD'] for b in fold_best_list])
    asds = np.array([b['ASD'] for b in fold_best_list])
    best_epochs = [b['epoch'] for b in fold_best_list]

    class_dices = np.array([b['class_Dice'] for b in fold_best_list])  # (F, C)
    class_ious = np.array([b['class_IOU'] for b in fold_best_list])    # (F, C)
    class_hds = np.array([b['class_HD'] for b in fold_best_list])      # (F, C)
    class_asds = np.array([b['class_ASD'] for b in fold_best_list])    # (F, C)

    print("\n" + "=" * 85)
    print(f"ΣΥΓΚΕΝΤΡΩΤΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ {n_folds}-FOLD CROSS-VALIDATION (mean ± std)")
    print("=" * 85)
    print(f"  Average Dice : {np.mean(dices):.4f} ± {np.std(dices):.4f}  ({100*np.mean(dices):.2f}% ± {100*np.std(dices):.2f}%)")
    print(f"  Average IoU  : {np.mean(ious):.4f} ± {np.std(ious):.4f}  ({100*np.mean(ious):.2f}% ± {100*np.std(ious):.2f}%)")
    print(f"  Average HD   : {np.mean(hds):.2f} ± {np.std(hds):.2f} px")
    print(f"  Average ASD  : {np.mean(asds):.2f} ± {np.std(asds):.2f} px")
    print("-" * 85)
    print("  ΑΝΑΛΥΣΗ ΑΝΑ ΑΝΑΤΟΜΙΚΗ ΔΟΜΗ (mean ± std):")
    for c in range(1, n_classes):
        c_name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"Class {c}"
        print(
            f"    [{c_name:<5}] "
            f"Dice: {np.mean(class_dices[:, c]):.4f} ± {np.std(class_dices[:, c]):.4f} | "
            f"IoU: {np.mean(class_ious[:, c]):.4f} ± {np.std(class_ious[:, c]):.4f} | "
            f"HD: {np.mean(class_hds[:, c]):.2f} ± {np.std(class_hds[:, c]):.2f}px | "
            f"ASD: {np.mean(class_asds[:, c]):.2f} ± {np.std(class_asds[:, c]):.2f}px"
        )
    print("-" * 85)
    rec_epochs = int(round(float(np.mean(best_epochs))))
    print(f"  Best Epochs ανά Fold : {best_epochs}")
    print(f"  Μέσος όρος Best Epoch: {np.mean(best_epochs):.1f} -> Προτεινόμενα epochs για `--mode final_train`: --epochs {rec_epochs}")
    print("=" * 85 + "\n")


def run_cv_mode(args, device):
    """
    Εκτελεί το 5-fold Cross-Validation στο 80% των δεδομένων.
    Σε κάθε fold:
      - Ορίζεται σταθερό seed (π.χ. 42 + fold -> 42, 43, 44, 45, 46)
      - Το μοντέλο & ο optimizer αρχικοποιούνται ΕΚ ΝΕΟΥ (χωρίς μεταφορά βαρών)
      - Εφαρμόζεται Data Augmentation ΜΟΝΟ στα 4 training folds
      - Εφαρμόζεται Early Stopping αποκλειστικά στο validation fold
    """
    folds_to_run = [args.fold] if args.fold is not None else [0, 1, 2, 3, 4]
    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)

    criterion = CEDiceLoss(
        ce_weight=1.0, dice_weight=1.0, class_weights=(0.5, 1.0, 15.0, 5.0)
    ).to(device)

    for idx, fold in enumerate(folds_to_run):
        existing_ckpt = os.path.join(models_dir, f'UNet_fold{fold}_best.pt')
        if args.skip_completed and os.path.isfile(existing_ckpt):
            print(f"\n[Skip] Το Fold {fold} έχει ήδη ολοκληρωθεί ({existing_ckpt}) -> Προσπερνάται.")
            continue

        fold_seed = args.seed + fold
        seed_everything(fold_seed)

        train_csv = os.path.join(args.splits_dir, f'fold_{fold}_train.csv')
        val_csv = os.path.join(args.splits_dir, f'fold_{fold}_val.csv')

        print("\n" + "#" * 80)
        print(f"ΕΚΚΙΝΗΣΗ FOLD {fold} / 4  (Seed = {fold_seed})")
        print(f"  Train CSV : {train_csv}")
        print(f"  Val CSV   : {val_csv}")
        print("#" * 80)

        # Πλήρης επαναρχικοποίηση μοντέλου και optimizer από το μηδέν για το τρέχον fold
        model, optimizer, model_dict = get_model(
            args, device, load_checkpoint=None, print_complexity=(idx == 0)
        )
        model_dict['fold'] = fold
        model_dict['seed'] = fold_seed

        train_loader = get_dataloader(
            csv_path=train_csv,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            augment=True,       # Data augmentation ΜΟΝΟ στο training set
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device == 'cuda'),
            seed=fold_seed
        )
        val_loader = get_dataloader(
            csv_path=val_csv,
            data_dir=args.data_dir,
            batch_size=args.test_batch_size,
            augment=False,      # ΠΟΤΕ augmentation στο validation set
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device == 'cuda'),
            seed=fold_seed
        )

        writer = SummaryWriter(log_dir=f'runs/fold_{fold}') if args.tensorboard else None
        checkpoint_path = os.path.join(models_dir, f'{model.name}_fold{fold}_best.pt')

        epochs_no_improve = 0

        for epoch in range(1, args.epochs + 1):
            train_loss = train(
                model, device, train_loader, optimizer, criterion, epoch, log_interval=args.log_interval
            )
            val_loss, val_iou, val_dice, val_hd, val_asd, val_c_iou, val_c_dice, val_c_hd, val_c_asd = validate(
                model, device, val_loader, criterion, args.n_classes, split_name=f"Fold {fold} Validation"
            )

            if writer is not None:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('IOU/val', val_iou, epoch)
                writer.add_scalar('Dice/val', val_dice, epoch)
                writer.add_scalar('HD/val', val_hd, epoch)
                writer.add_scalar('ASD/val', val_asd, epoch)
                for c in range(1, args.n_classes):
                    c_name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f'Class_{c}'
                    writer.add_scalar(f'IOU/{c_name}', val_c_iou[c], epoch)
                    writer.add_scalar(f'Dice/{c_name}', val_c_dice[c], epoch)
                    writer.add_scalar(f'HD/{c_name}', val_c_hd[c], epoch)
                    writer.add_scalar(f'ASD/{c_name}', val_c_asd[c], epoch)

            model_dict['train_loss'].append(train_loss)
            model_dict['test_loss'].append(val_loss)
            model_dict['metrics']['IOU'].append(val_iou)
            model_dict['metrics']['class_IOU'].append(val_c_iou)
            model_dict['metrics']['Dice'].append(val_dice)
            model_dict['metrics']['class_Dice'].append(val_c_dice)
            model_dict['metrics']['HD'].append(val_hd)
            model_dict['metrics']['class_HD'].append(val_c_hd)
            model_dict['metrics']['ASD'].append(val_asd)
            model_dict['metrics']['class_ASD'].append(val_c_asd)

            # Αποθήκευση καλύτερου μοντέλου αποκλειστικά βάσει Validation Dice
            if epoch == 1 or val_dice > model_dict['metrics']['best']['Dice']:
                model_dict['model_state_dict'] = copy.deepcopy(model.state_dict())
                model_dict['optimizer_state_dict'] = copy.deepcopy(optimizer.state_dict())
                model_dict['metrics']['best']['IOU'] = val_iou
                model_dict['metrics']['best']['class_IOU'] = val_c_iou
                model_dict['metrics']['best']['Dice'] = val_dice
                model_dict['metrics']['best']['class_Dice'] = val_c_dice
                model_dict['metrics']['best']['HD'] = val_hd
                model_dict['metrics']['best']['class_HD'] = val_c_hd
                model_dict['metrics']['best']['ASD'] = val_asd
                model_dict['metrics']['best']['class_ASD'] = val_c_asd
                model_dict['metrics']['best']['epoch'] = epoch
                epochs_no_improve = 0

                if args.save:
                    torch.save(model_dict, checkpoint_path)
                    print(f"  [Checkpoint] Νέο καλύτερο Validation Dice ({val_dice:.4f}) στο Epoch {epoch} -> Αποθηκεύτηκε στο {checkpoint_path}")
            else:
                epochs_no_improve += 1
                print(
                    f"  [EarlyStopping] Καμία βελτίωση στο Validation Dice για {epochs_no_improve}/{args.patience} epochs "
                    f"(Best: {model_dict['metrics']['best']['Dice']:.4f} @ Epoch {model_dict['metrics']['best']['epoch']})"
                )
                if args.patience > 0 and epochs_no_improve >= args.patience:
                    print(f"\n>>> Early Stopping ενεργοποιήθηκε στο Fold {fold} (Epoch {epoch})! <<<")
                    break

        if writer is not None:
            writer.close()

        print_best_summary(
            model.name,
            model_dict['metrics']['best'],
            args.n_classes,
            complexity=model_dict.get('complexity'),
            header=f"ΚΑΛΥΤΕΡΑ ΑΠΟΤΕΛΕΣΜΑΤΑ FOLD {fold}"
        )

    # Αν υπάρχουν αποθηκευμένα και τα 5 folds στον φάκελο models/, τυπώνουμε το συνολικό mean ± std
    all_fold_bests = []
    for k in range(5):
        ckpt_file = os.path.join(models_dir, f'UNet_fold{k}_best.pt')
        if os.path.isfile(ckpt_file):
            ckpt_data = torch.load(ckpt_file, map_location='cpu', weights_only=False)
            all_fold_bests.append(ckpt_data['metrics']['best'])

    if len(all_fold_bests) == 5:
        print_cv_summary_across_folds(all_fold_bests, args.n_classes)


def run_final_train_mode(args, device):
    """
    Εκπαιδεύει εκ νέου το μοντέλο από την αρχή σε ΟΛΟΚΛΗΡΟ το 80% Development Set (`dev_80.csv`)
    για τον αριθμό των epochs (`--epochs`) που καθορίστηκε από το 5-Fold Cross-Validation,
    χωρίς καμία χρήση του ανεξάρτητου Test Set.
    """
    seed_everything(args.seed)
    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)

    dev_csv = os.path.join(args.splits_dir, 'dev_80.csv')
    print("\n" + "#" * 80)
    print(f"ΤΕΛΙΚΗ ΕΚΠΑΙΔΕΥΣΗ ΣΕ ΟΛΟΚΛΗΡΟ ΤΟ 80% DEVELOPMENT SET (Epochs = {args.epochs}, Seed = {args.seed})")
    print(f"  Dev 80% CSV: {dev_csv}")
    print("#" * 80)

    model, optimizer, model_dict = get_model(args, device, load_checkpoint=None, print_complexity=True)

    train_loader = get_dataloader(
        csv_path=dev_csv,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        augment=True,           # Data augmentation στο 80% training set
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda'),
        seed=args.seed
    )

    criterion = CEDiceLoss(
        ce_weight=1.0, dice_weight=1.0, class_weights=(0.5, 1.0, 15.0, 5.0)
    ).to(device)

    for epoch in range(1, args.epochs + 1):
        train_loss = train(
            model, device, train_loader, optimizer, criterion, epoch, log_interval=args.log_interval
        )
        model_dict['train_loss'].append(train_loss)
        print(f"Epoch {epoch}/{args.epochs} ολοκληρώθηκε -> Μέσο Train Loss: {train_loss:.4f}")

    model_dict['total_epoch'] = args.epochs
    model_dict['model_state_dict'] = copy.deepcopy(model.state_dict())
    model_dict['optimizer_state_dict'] = copy.deepcopy(optimizer.state_dict())

    final_ckpt_path = os.path.join(models_dir, f'{model.name}_final_dev80.pt')
    torch.save(model_dict, final_ckpt_path)
    print("\n" + "=" * 80)
    print(f"Η τελική εκπαίδευση στο 80% ολοκληρώθηκε επιτυχώς!")
    print(f"Το τελικό μοντέλο αποθηκεύτηκε στο: {final_ckpt_path}")
    print(f"Για την εφάπαξ αξιολόγηση στο ανεξάρτητο 20% Test Set, τρέξε με: --mode test")
    print("=" * 80 + "\n")


def run_test_mode(args, device):
    """
    Αξιολογεί το τελικό εκπαιδευμένο μοντέλο του 80% ΜΙΑ ΦΟΡΑ
    πάνω στο ανεξάρτητο 20% Test Set (`test.csv`).
    """
    seed_everything(args.seed)
    ckpt_name = args.model if args.model is not None else 'UNet_final_dev80.pt'
    test_csv = os.path.join(args.splits_dir, 'test.csv')

    print("\n" + "#" * 80)
    print(f"ΤΕΛΙΚΗ ΑΞΙΟΛΟΓΗΣΗ ΣΤΟ ΑΝΕΞΑΡΤΗΤΟ 20% TEST SET")
    print(f"  Checkpoint : models/{ckpt_name}")
    print(f"  Test CSV   : {test_csv}")
    print("#" * 80)

    model, _, model_dict = get_model(args, device, load_checkpoint=ckpt_name, print_complexity=True)

    test_loader = get_dataloader(
        csv_path=test_csv,
        data_dir=args.data_dir,
        batch_size=args.test_batch_size,
        augment=False,          # ΠΟΤΕ augmentation στο test set
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda'),
        seed=args.seed
    )

    criterion = CEDiceLoss(
        ce_weight=1.0, dice_weight=1.0, class_weights=(0.5, 1.0, 15.0, 5.0)
    ).to(device)

    test_loss, test_iou, test_dice, test_hd, test_asd, test_c_iou, test_c_dice, test_c_hd, test_c_asd = validate(
        model, device, test_loader, criterion, args.n_classes, split_name="Independent Test (20%)"
    )

    test_metrics = {
        'epoch': model_dict.get('total_epoch', 0),
        'IOU': test_iou,
        'class_IOU': test_c_iou,
        'Dice': test_dice,
        'class_Dice': test_c_dice,
        'HD': test_hd,
        'class_HD': test_c_hd,
        'ASD': test_asd,
        'class_ASD': test_c_asd,
    }
    print_best_summary(
        model.name,
        test_metrics,
        args.n_classes,
        complexity=model_dict.get('complexity'),
        header="ΤΕΛΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ ΑΝΕΞΑΡΤΗΤΟΥ TEST SET (20%)"
    )


if __name__ == '__main__':
    args = parse_args()

    device = ('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    gpu_info = f" ({torch.cuda.get_device_name(0)})" if device == 'cuda' else ""
    print(f"Συσκευή Εκτέλεσης: {device}{gpu_info}")
    print(f"Φάκελος Dataset  : {args.data_dir}")
    print(f"Φάκελος Splits   : {args.splits_dir}")
    print(f"Mode Εκτέλεσης   : {args.mode}")

    if args.mode == 'cv':
        run_cv_mode(args, device)
    elif args.mode == 'final_train':
        run_final_train_mode(args, device)
    elif args.mode == 'test':
        run_test_mode(args, device)
