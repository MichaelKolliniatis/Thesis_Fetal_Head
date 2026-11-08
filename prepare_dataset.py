"""
Prepare Dataset Splits (80% Dev / 20% Test + 5-Fold Cross-Validation)
=====================================================================
Υλοποιεί τις οδηγίες διαχωρισμού της βάσης για τη διπλωματική εργασία:
1. Εντοπίζει τις 3.832 εικόνες και τις αντίστοιχες καθαρές μάσκες (`masks/`)
   στους 4 φακέλους κατηγοριών (`~/Dataset/1` έως `~/Dataset/4`).
2. Εξάγει το `patient_id` από το όνομα κάθε αρχείου:
   - Φάκελος 1 (Diverse / HC18): `014_2HC.png` -> `HC18_014`
   - Φάκελοι 2, 3, 4 (BCNatal):  `Patient00188_Plane3_1_of_3.png` -> `BCN_Patient00188`
3. Με `random_seed = 42` και `StratifiedGroupKFold(n_splits=5)`:
   - Κρατάει το 20% ως ανεξάρτητο Test Set (`test.csv`) και το 80% ως Development Set (`dev_80.csv`),
     διατηρώντας την αναλογία των 4 κατηγοριών (stratification) και μηδενική επικάλυψη ασθενών.
   - Χωρίζει το 80% σε 5 folds (`fold_0_train.csv`, `fold_0_val.csv` ... `fold_4_train.csv`, `fold_4_val.csv`).
4. Εκτελεί αυστηρούς ελέγχους (`assert`) για μηδενικό Data Leakage και τυπώνει
   αναλυτικό πίνακα κατανομής για το κείμενο της διπλωματικής.
"""

import os
import re
import csv
import glob
import argparse
import numpy as np


RANDOM_SEED = 42

CATEGORY_MAP = {
    '1': 'Diverse',
    '2': 'Trans-cerebellum',
    '3': 'Trans-thalamic',
    '4': 'Trans-ventricular',
}


def stratified_group_k_fold_indices(y, groups, n_splits=5, seed=42):
    """
    Χωρίζει τα δείγματα σε `n_splits` folds με ταυτόχρονο Stratification ως προς `y`
    και ομαδοποίηση ως προς `groups` (ώστε κανένα group/patient_id να μην σπάει σε >1 fold).
    Χρησιμοποιεί το επίσημο `StratifiedGroupKFold` του scikit-learn αν είναι εγκατεστημένο,
    αλλιώς ισοδύναμη υλοποίηση σε καθαρό NumPy (ώστε να μην αποτύχει ποτέ αν λείπει το sklearn).
    """
    y_arr = np.asarray(y)
    groups_arr = np.asarray(groups)

    try:
        from sklearn.model_selection import StratifiedGroupKFold
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(sgkf.split(np.zeros(len(y_arr)), y_arr, groups_arr))
    except ImportError:
        pass

    # Fallback υλοποίηση (ίδιος αλγόριθμος ελαχιστοποίησης διακύμανσης του scikit-learn)
    rng = np.random.RandomState(seed)
    unique_classes, y_inversed = np.unique(y_arr, return_inverse=True)
    n_classes = len(unique_classes)

    unique_groups, group_inv = np.unique(groups_arr, return_inverse=True)
    n_groups = len(unique_groups)

    y_counts_per_group = np.zeros((n_groups, n_classes), dtype=np.int64)
    for cls_idx, grp_idx in zip(y_inversed, group_inv):
        y_counts_per_group[grp_idx, cls_idx] += 1

    y_counts_total = np.sum(y_counts_per_group, axis=0)

    # Ανακάτεμα με σταθερό seed και ταξινόμηση ομάδων κατά τυπική απόκλιση (μεγαλύτερες πρώτα)
    shuffled_order = np.arange(n_groups)
    rng.shuffle(shuffled_order)
    stds = np.std(y_counts_per_group[shuffled_order] / np.maximum(y_counts_total, 1), axis=1)
    sorted_group_indices = shuffled_order[np.argsort(-stds, kind='mergesort')]

    y_counts_per_fold = np.zeros((n_splits, n_classes), dtype=np.int64)
    group_to_fold = np.full(n_groups, -1, dtype=np.int64)

    for grp_idx in sorted_group_indices:
        grp_counts = y_counts_per_group[grp_idx]
        best_fold = -1
        min_eval = None
        for k in range(n_splits):
            y_counts_per_fold[k] += grp_counts
            fold_stds = np.std(y_counts_per_fold / np.maximum(y_counts_total, 1), axis=0)
            score = np.mean(fold_stds)
            y_counts_per_fold[k] -= grp_counts
            if min_eval is None or score < min_eval:
                min_eval = score
                best_fold = k
        y_counts_per_fold[best_fold] += grp_counts
        group_to_fold[grp_idx] = best_fold

    sample_folds = group_to_fold[group_inv]
    splits = []
    all_indices = np.arange(len(y_arr))
    for k in range(n_splits):
        val_idx = all_indices[sample_folds == k]
        train_idx = all_indices[sample_folds != k]
        splits.append((train_idx, val_idx))
    return splits


def extract_patient_id(filename, cat_key):
    """
    Εξάγει το μοναδικό ID ασθενούς/εξέτασης από το όνομα του αρχείου εικόνας.
    """
    if cat_key == '1':
        # Μορφή HC18: π.χ. 000_HC.png, 014_2HC.png, 014_3HC.png -> HC18_014
        m = re.match(r'^(\d+)_', filename)
        if m:
            return f"HC18_{m.group(1)}"
        return f"HC18_{os.path.splitext(filename)[0]}"
    else:
        # Μορφή BCNatal: π.χ. Patient00188_Plane3_1_of_3.png -> BCN_Patient00188
        m = re.match(r'^(Patient\d+)_', filename)
        if m:
            return f"BCN_{m.group(1)}"
        return f"BCN_{os.path.splitext(filename)[0]}"


def load_pixel_sizes(category_dir):
    """
    Διαβάζει το αρχείο `*-Pixel-Size.csv` (αν υπάρχει στον φάκελο κατηγορίας)
    και επιστρέφει λεξικό {filename: pixel_size_str}.
    """
    pixel_map = {}
    csv_files = glob.glob(os.path.join(category_dir, '*.csv'))
    for csv_path in csv_files:
        with open(csv_path, mode='r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    label = row[0].strip()
                    val = row[1].strip()
                    if label:
                        pixel_map[label] = val
    return pixel_map


def find_images_subdir(category_dir, masks_subdir='masks'):
    """
    Εντοπίζει αυτόματα τον υποφάκελο με τις εικόνες υπερήχων (.png) μέσα στην κατηγορία.
    """
    valid_exts = ('.png', '.jpg', '.jpeg')
    best_dir = None
    best_count = 0

    for entry in sorted(os.listdir(category_dir)):
        full_path = os.path.join(category_dir, entry)
        if not os.path.isdir(full_path):
            continue
        if entry == masks_subdir or 'cvat' in entry.lower() or 'segmentation' in entry.lower():
            continue

        img_count = sum(1 for f in os.listdir(full_path) if f.lower().endswith(valid_exts))
        if img_count > best_count:
            best_count = img_count
            best_dir = full_path

    return best_dir


def collect_all_records(dataset_dir, masks_subdir='masks'):
    """
    Σαρώνει τους 4 φακέλους κατηγοριών, επαληθεύει ότι για κάθε εικόνα υπάρχει
    η αντίστοιχη μάσκα στο `masks/`, και επιστρέφει τη λίστα εγγραφών.
    """
    records = []
    valid_exts = ('.png', '.jpg', '.jpeg')

    for entry in sorted(os.listdir(dataset_dir)):
        cat_dir = os.path.join(dataset_dir, entry)
        if not os.path.isdir(cat_dir) or entry[0] not in CATEGORY_MAP:
            continue

        cat_key = entry[0]
        cat_name = CATEGORY_MAP[cat_key]

        images_dir = find_images_subdir(cat_dir, masks_subdir=masks_subdir)
        masks_dir = os.path.join(cat_dir, masks_subdir)

        if images_dir is None or not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Δεν βρέθηκε φάκελος εικόνων μέσα στο: {cat_dir}")
        if not os.path.isdir(masks_dir):
            raise FileNotFoundError(
                f"Δεν βρέθηκε ο φάκελος μασκών `{masks_dir}`. "
                f"Τρέξε πρώτα το `prepare_masks.py`!"
            )

        pixel_map = load_pixel_sizes(cat_dir)
        image_files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(valid_exts))

        for img_name in image_files:
            img_abs = os.path.join(images_dir, img_name)
            msk_abs = os.path.join(masks_dir, img_name)
            if not os.path.isfile(msk_abs):
                raise FileNotFoundError(
                    f"Λείπει η μάσκα για την εικόνα {img_name}: {msk_abs}"
                )

            # Σχετικά μονοπάτια (με '/' ώστε να είναι 100% συμβατά σε Linux/Windows)
            img_rel = os.path.relpath(img_abs, dataset_dir).replace('\\', '/')
            msk_rel = os.path.relpath(msk_abs, dataset_dir).replace('\\', '/')

            patient_id = extract_patient_id(img_name, cat_key)
            pixel_mm = pixel_map.get(img_name, '')

            records.append({
                'image_id': img_name,
                'category': cat_name,
                'category_folder': entry,
                'patient_id': patient_id,
                'image_rel_path': img_rel,
                'mask_rel_path': msk_rel,
                'pixel_size_mm': pixel_mm,
                'split': '',
                'fold': -1,
            })

    return records


def save_csv(filepath, rows, fieldnames):
    """Αποθηκεύει λίστα λεξικών σε αρχείο CSV."""
    with open(filepath, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary_table(title, subset_rows):
    """Τυπώνει αναλυτικά στατιστικά ενός συνόλου (εικόνες, ασθενείς, αναλογία κατηγοριών)."""
    n_total = len(subset_rows)
    n_patients = len(set(r['patient_id'] for r in subset_rows))
    cat_counts = {cat: 0 for cat in CATEGORY_MAP.values()}
    for r in subset_rows:
        cat_counts[r['category']] += 1

    cat_str = " | ".join(
        f"{cat}: {cnt:4d} ({100.0 * cnt / max(n_total, 1):5.1f}%)"
        for cat, cnt in cat_counts.items()
    )
    print(f"  {title:<18} -> Εικόνες: {n_total:4d} | Ασθενείς: {n_patients:4d} || {cat_str}")


def main():
    parser = argparse.ArgumentParser(
        description='Δημιουργία CSV Splits (80/20 + 5-Fold StratifiedGroupKFold CV)'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default=os.path.expanduser('~/Dataset'),
        help='Μονοπάτι του φακέλου Dataset (προεπιλογή: ~/Dataset)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Φάκελος αποθήκευσης των αρχείων CSV (προεπιλογή: <dataset-dir>/splits, δηλ. ~/Dataset/splits)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=RANDOM_SEED,
        help='Random seed για αναπαραγωγιμότητα (προεπιλογή: 42)'
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(args.dataset_dir, 'splits')

    print("=" * 105)
    print(f"Δημιουργία Dataset Splits με StratifiedGroupKFold (random_seed = {args.seed})")
    print(f"Φάκελος Dataset : {os.path.abspath(args.dataset_dir)}")
    print(f"Φάκελος εξόδου  : {os.path.abspath(args.output_dir)}")
    print("=" * 105)

    records = collect_all_records(args.dataset_dir)
    n_total = len(records)
    if n_total == 0:
        raise RuntimeError("Δεν βρέθηκαν εικόνες στη βάση!")

    y_all = np.array([r['category'] for r in records])
    groups_all = np.array([r['patient_id'] for r in records])

    # ΒΗΜΑ 1: Διαχωρισμός 80% Development (Train+Val) / 20% Ανεξάρτητο Test Set
    outer_splits = stratified_group_k_fold_indices(
        y_all, groups_all, n_splits=5, seed=args.seed
    )
    dev_indices, test_indices = outer_splits[0]

    for idx in test_indices:
        records[idx]['split'] = 'test'
        records[idx]['fold'] = -1

    # ΒΗΜΑ 2: 5-Fold Cross-Validation μέσα στο 80% (Development Set)
    y_dev = y_all[dev_indices]
    groups_dev = groups_all[dev_indices]

    inner_splits = stratified_group_k_fold_indices(
        y_dev, groups_dev, n_splits=5, seed=args.seed
    )

    for fold_idx, (train_sub_idx, val_sub_idx) in enumerate(inner_splits):
        for sub_i in val_sub_idx:
            orig_idx = dev_indices[sub_i]
            records[orig_idx]['split'] = 'dev'
            records[orig_idx]['fold'] = fold_idx

    # ΒΗΜΑ 3: Αυστηροί έλεγχοι (Sanity Checks) για μηδενικό Data Leakage
    dev_records = [r for r in records if r['split'] == 'dev']
    test_records = [r for r in records if r['split'] == 'test']

    dev_patients = set(r['patient_id'] for r in dev_records)
    test_patients = set(r['patient_id'] for r in test_records)
    overlap_dev_test = dev_patients & test_patients
    assert len(overlap_dev_test) == 0, f"DATA LEAKAGE μεταξύ Dev και Test: {overlap_dev_test}"
    assert len(dev_records) + len(test_records) == n_total, "Ασυμφωνία συνολικού πλήθους εικόνων!"

    # ΒΗΜΑ 4: Αποθήκευση όλων των CSV αρχείων στον φάκελο splits/
    os.makedirs(args.output_dir, exist_ok=True)
    fieldnames = [
        'image_id',
        'category',
        'category_folder',
        'patient_id',
        'image_rel_path',
        'mask_rel_path',
        'pixel_size_mm',
        'split',
        'fold',
    ]

    save_csv(os.path.join(args.output_dir, 'master_splits.csv'), records, fieldnames)
    save_csv(os.path.join(args.output_dir, 'dev_80.csv'), dev_records, fieldnames)
    save_csv(os.path.join(args.output_dir, 'test.csv'), test_records, fieldnames)

    for k in range(5):
        fold_train = [r for r in dev_records if r['fold'] != k]
        fold_val = [r for r in dev_records if r['fold'] == k]

        # Έλεγχος μηδενικής επικάλυψης ασθενών μεταξύ Train και Val στο Fold k
        p_train = set(r['patient_id'] for r in fold_train)
        p_val = set(r['patient_id'] for r in fold_val)
        assert len(p_train & p_val) == 0, f"DATA LEAKAGE στο Fold {k}!"
        assert len(p_val & test_patients) == 0, f"DATA LEAKAGE μεταξύ Fold {k} και Test!"

        save_csv(os.path.join(args.output_dir, f'fold_{k}_train.csv'), fold_train, fieldnames)
        save_csv(os.path.join(args.output_dir, f'fold_{k}_val.csv'), fold_val, fieldnames)

    # ΒΗΜΑ 5: Εκτύπωση Αναφοράς & Πίνακα Κατανομής
    print("\n1. ΚΕΝΤΡΙΚΟΣ ΔΙΑΧΩΡΙΣΜΟΣ (100% -> 80% DEV / 20% TEST)")
    print("-" * 105)
    print_summary_table("Total (100%)", records)
    print_summary_table("Dev Set (80%)", dev_records)
    print_summary_table("Test Set (20%)", test_records)

    print("\n2. ΚΑΤΑΝΟΜΗ 5-FOLD CROSS-VALIDATION (ΜΕΣΑ ΣΤΟ 80% DEV SET)")
    print("-" * 105)
    for k in range(5):
        fold_train = [r for r in dev_records if r['fold'] != k]
        fold_val = [r for r in dev_records if r['fold'] == k]
        print_summary_table(f"Fold {k} Train", fold_train)
        print_summary_table(f"Fold {k} Val", fold_val)
        print("-" * 105)

    print("\n3. ΕΛΕΓΧΟΣ ΕΓΚΥΡΟΤΗΤΑΣ (DATA LEAKAGE VERIFICATION)")
    print(f"  - Κοινοί ασθενείς μεταξύ Dev (80%) και Test (20%) : {len(overlap_dev_test)} (OK - Μηδενικό Data Leakage)")
    print(f"  - Κοινοί ασθενείς μεταξύ Train και Val στα Folds  : 0 σε όλα τα Folds 0..4 (OK)")
    print(f"  - Αποθηκεύτηκαν επιτυχώς 13 αρχεία CSV στον φάκελο: {os.path.abspath(args.output_dir)}/")
    print("=" * 105)


if __name__ == '__main__':
    main()
