import torch
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion

def dice_coef(outputs, targets, n_classes):
    """
    Επιστρέφει το Intersection και το Union αντί για το τελικό κλάσμα, 
    ώστε να υπολογιστεί το Global Dice χωρίς πλασματικά '1.0' σε άδειες κλάσεις.
    """
    batch_size = outputs.shape[0]
    intersection = np.zeros(n_classes)
    union = np.zeros(n_classes)

    # Ξεκινάμε από το 1 για να αγνοήσουμε εντελώς το μαύρο φόντο (κλάση 0)
    for c in range(1, n_classes):
        out_c = (outputs == c).reshape(batch_size, -1).float()
        tgt_c = (targets == c).reshape(batch_size, -1).float()

        intersection[c] = (out_c * tgt_c).sum().item()
        union[c] = (out_c.sum() + tgt_c.sum()).item()

    return intersection, union

def iou(outputs, targets, n_classes):
    """
    Επιστρέφει την Τομή (Intersection) και την Ένωση (Union) ανά κλάση,
    ώστε να υπολογιστεί το Global IoU χωρίς πλασματικά '1.0' σε άδειες κλάσεις.
    """
    batch_size = outputs.shape[0]
    intersection = np.zeros(n_classes)
    union = np.zeros(n_classes)

    # Ξεκινάμε από το 1 για να αγνοήσουμε εντελώς το μαύρο φόντο (κλάση 0)
    for c in range(1, n_classes):
        out_c = (outputs == c).reshape(batch_size, -1)
        tgt_c = (targets == c).reshape(batch_size, -1)

        intersection[c] = (out_c & tgt_c).sum().item()
        union[c] = (out_c | tgt_c).sum().item()

    return intersection, union

def hausdorff_distance(outputs, targets, n_classes, h=256, w=256):
    """
    Επιστρέφει το άθροισμα των αποστάσεων Hausdorff (hd_sum) και το πλήθος
    των έγκυρων εικόνων (hd_count) ανά κλάση, αγνοώντας τις τομές όπου 
    ένα όργανο απουσιάζει τόσο από την πραγματική μάσκα όσο και από την πρόβλεψη
    (ώστε να μην μπαίνουν πλασματικά '0.0px' σε άδειες τομές).
    """
    batch_size = outputs.shape[0]
    hd_sum = np.zeros(n_classes)
    hd_count = np.zeros(n_classes)
    max_dist = np.sqrt(h**2 + w**2)

    # Επαναφορά της 1D λίστας σε 2D εικόνες (B, 256, 256) για να έχουν χωρικό νόημα οι αποστάσεις
    out_np = outputs.detach().cpu().numpy().reshape(batch_size, h, w)
    tgt_np = targets.detach().cpu().numpy().reshape(batch_size, h, w)

    for idx in range(batch_size):
        # Υπολογισμός του HD μόνο για τα όργανα (κλάσεις 1..n_classes-1), αγνοώντας το μαύρο φόντο (0)
        for c in range(1, n_classes):
            out_mask = (out_np[idx] == c)
            tgt_mask = (tgt_np[idx] == c)

            has_out = out_mask.any()
            has_tgt = tgt_mask.any()

            if not has_out and not has_tgt:
                # Και τα δύο είναι άδεια -> δεν υπάρχει περίγραμμα προς μέτρηση σε αυτή την τομή
                continue
            elif not has_out or not has_tgt:
                # Το ένα είναι άδειο και το άλλο όχι -> μέγιστη δυνατή απόσταση (διαγώνιος)
                hd = max_dist
            else:
                # Εξαγωγή περιγραμμάτων (contours) και αμφίδρομη μέγιστη απόσταση (Hausdorff)
                out_surface = out_mask & ~binary_erosion(out_mask)
                tgt_surface = tgt_mask & ~binary_erosion(tgt_mask)

                dist_to_tgt = distance_transform_edt(~tgt_surface)
                dist_to_out = distance_transform_edt(~out_surface)
                hd1 = dist_to_tgt[out_surface].max()
                hd2 = dist_to_out[tgt_surface].max()
                hd = float(max(hd1, hd2))

            hd_sum[c] += hd
            hd_count[c] += 1

    return hd_sum, hd_count

def average_surface_distance(outputs, targets, n_classes, h=256, w=256):
    """
    Υπολογίζει το Average Symmetric Surface Distance (ASD / ASSD).
    Επιστρέφει το άθροισμα των μέσων αποστάσεων περιγράμματος (asd_sum) και το πλήθος
    των έγκυρων εικόνων (asd_count) ανά κλάση, αγνοώντας τις διπλά άδειες τομές.
    """
    batch_size = outputs.shape[0]
    asd_sum = np.zeros(n_classes)
    asd_count = np.zeros(n_classes)
    max_dist = np.sqrt(h**2 + w**2)

    out_np = outputs.detach().cpu().numpy().reshape(batch_size, h, w)
    tgt_np = targets.detach().cpu().numpy().reshape(batch_size, h, w)

    for idx in range(batch_size):
        for c in range(1, n_classes):
            out_mask = (out_np[idx] == c)
            tgt_mask = (tgt_np[idx] == c)

            has_out = out_mask.any()
            has_tgt = tgt_mask.any()

            if not has_out and not has_tgt:
                continue
            elif not has_out or not has_tgt:
                asd = max_dist
            else:
                # Εξαγωγή των pixels περιγράμματος μέσω μορφολογικής διάβρωσης
                out_surface = out_mask & ~binary_erosion(out_mask)
                tgt_surface = tgt_mask & ~binary_erosion(tgt_mask)

                dist_to_tgt = distance_transform_edt(~tgt_surface)
                dist_to_out = distance_transform_edt(~out_surface)

                d_out_to_tgt = dist_to_tgt[out_surface]
                d_tgt_to_out = dist_to_out[tgt_surface]

                asd = float(
                    (d_out_to_tgt.sum() + d_tgt_to_out.sum())
                    / (d_out_to_tgt.size + d_tgt_to_out.size)
                )

            asd_sum[c] += asd
            asd_count[c] += 1

    return asd_sum, asd_count


def compute_model_complexity(model, device, input_size=(1, 1, 256, 256)):
    """
    Υπολογίζει τα υπολογιστικά & μηχανολογικά χαρακτηριστικά του μοντέλου:
      - total_params: Συνολικές παράμετροι
      - trainable_params: Εκπαιδεύσιμες παράμετροι (requires_grad=True)
      - model_size_mb: Αποτύπωμα μνήμης βαρών & buffers σε MB (π.χ. float32)
      - gflops: Δισεκατομμύρια πράξεις κινητής υποδιαστολής (GFLOPs) για 1 εικόνα (1, 1, 256, 256)
    Εκτελείται μία φορά στην εκκίνηση χωρίς να επηρεάζει την κατάσταση του μοντέλου.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_bytes + buffer_bytes) / (1024 ** 2)

    gflops = None
    was_training = model.training
    try:
        from torch.utils.flop_counter import FlopCounterMode

        model.eval()
        dummy_input = torch.randn(*input_size, device=device)
        flop_counter = FlopCounterMode(display=False)
        with torch.no_grad(), flop_counter:
            _ = model(dummy_input)
        gflops = flop_counter.get_total_flops() / 1e9
    except Exception:
        gflops = None
    finally:
        if was_training:
            model.train()

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size_mb,
        'gflops': gflops
    }