"""
Prepare Clean Segmentation Masks from CVAT XML Annotations
==========================================================
Διαβάζει τα πρωτογενή αρχεία `annotations.xml` (CVAT 1.1 format) και για τις 4
κατηγορίες της βάσης (`~/Dataset/1`, `~/Dataset/2`, `~/Dataset/3`, `~/Dataset/4`)
και παράγει καθαρές RGB μάσκες κατάτμησης στους φακέλους `masks/` κάθε κατηγορίας.

Τι διορθώνει σε σχέση με τις έτοιμες PNG μάσκες της βάσης:
1. Εξαλείφει το πρόβλημα επικάλυψης (`z_order` bug), σχεδιάζοντας ΠΑΝΤΑ πρώτα
   την έλλειψη του `Brain` και από πάνω τα `CSP` και `LV` (ανακτώντας και τις
   210 μάσκες CSP/LV που είχαν σβηστεί στις έτοιμες PNG μάσκες).
2. Ενοποιεί την χρωματική παλέτα RGB και στους 4 φακέλους ώστε να ταιριάζει
   απόλυτα με το `dataloader.py`:
     - Background (0): Μαύρο   (0, 0, 0)
     - Brain      (1): Κόκκινο (255, 0, 0)
     - CSP        (2): Κίτρινο (255, 255, 0)
     - LV         (3): Μπλε    (0, 0, 255)
"""

import os
import glob
import zipfile
import argparse
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image


# Ενιαία παλέτα RGB (συμβατή 100% με το dataloader.py)
CLASS_COLORS = {
    'Background': (0, 0, 0),
    'Brain': (255, 0, 0),       # Κλάση 1: Κόκκινο
    'CSP': (255, 255, 0),       # Κλάση 2: Κίτρινο (ενιαίο και στους 4 φακέλους)
    'LV': (0, 0, 255),          # Κλάση 3: Μπλε
}

# Σειρά σχεδίασης: Πρώτα το Brain (επίπεδο 1) και ΠΑΝΤΑ από πάνω τα CSP & LV (επίπεδο 2, 3)
DRAW_PRIORITY = {
    'Brain': 1,
    'CSP': 2,
    'LV': 3,
}


def draw_rotated_ellipse(mask_rgb, cx, cy, rx, ry, rotation_deg, color):
    """
    Σχεδιάζει γεμάτη περιστραμμένη έλλειψη (CVAT <ellipse>) πάνω στον πίνακα mask_rgb.
    Στο CVAT η γωνία rotation_deg είναι δεξιόστροφη στο σύστημα συντεταγμένων εικόνας.
    """
    if rx <= 0 or ry <= 0:
        return

    h, w, _ = mask_rgb.shape
    max_r = max(rx, ry) + 2.0
    x0 = max(0, int(np.floor(cx - max_r)))
    x1 = min(w, int(np.ceil(cx + max_r)) + 1)
    y0 = max(0, int(np.floor(cy - max_r)))
    y1 = min(h, int(np.ceil(cy + max_r)) + 1)

    if x0 >= x1 or y0 >= y1:
        return

    yy, xx = np.ogrid[y0:y1, x0:x1]
    dx = xx - cx
    dy = yy - cy

    theta = np.deg2rad(rotation_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    xr = dx * cos_t + dy * sin_t
    yr = -dx * sin_t + dy * cos_t

    inside = ((xr / rx) ** 2 + (yr / ry) ** 2) <= 1.0
    sub_mask = mask_rgb[y0:y1, x0:x1]
    sub_mask[inside] = color


def draw_rotated_box(mask_rgb, xtl, ytl, xbr, ybr, rotation_deg, color):
    """
    Σχεδιάζει γεμάτο περιστραμμένο ορθογώνιο (CVAT <box>) πάνω στον πίνακα mask_rgb.
    Το ορθογώνιο περιστρέφεται γύρω από το κέντρο του (cx, cy) κατά rotation_deg.
    """
    cx = 0.5 * (xtl + xbr)
    cy = 0.5 * (ytl + ybr)
    hw = 0.5 * abs(xbr - xtl)
    hh = 0.5 * abs(ybr - ytl)

    if hw <= 0 or hh <= 0:
        return

    h, w, _ = mask_rgb.shape
    diag = np.hypot(hw, hh) + 2.0
    x0 = max(0, int(np.floor(cx - diag)))
    x1 = min(w, int(np.ceil(cx + diag)) + 1)
    y0 = max(0, int(np.floor(cy - diag)))
    y1 = min(h, int(np.ceil(cy + diag)) + 1)

    if x0 >= x1 or y0 >= y1:
        return

    yy, xx = np.ogrid[y0:y1, x0:x1]
    dx = xx - cx
    dy = yy - cy

    theta = np.deg2rad(rotation_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    xr = dx * cos_t + dy * sin_t
    yr = -dx * sin_t + dy * cos_t

    inside = (np.abs(xr) <= hw) & (np.abs(yr) <= hh)
    sub_mask = mask_rgb[y0:y1, x0:x1]
    sub_mask[inside] = color


def find_annotations_xml(category_dir):
    """
    Εντοπίζει αυτόματα το `annotations.xml` μέσα στον φάκελο κατηγορίας:
      1. Απευθείας: `<category_dir>/annotations.xml` (π.χ. στους φακέλους 3 & 4)
      2. Σε υποφάκελο: `<category_dir>/*CVAT*/annotations.xml` (π.χ. στους φακέλους 1 & 2)
      3. Σε αρχείο zip: `<category_dir>/*CVAT*.zip` (αν δεν έχει αποσυμπιεστεί)
    Επιστρέφει (xml_root, source_description).
    """
    direct_xml = os.path.join(category_dir, 'annotations.xml')
    if os.path.isfile(direct_xml):
        return ET.parse(direct_xml).getroot(), direct_xml

    sub_xmls = glob.glob(os.path.join(category_dir, '*', 'annotations.xml'))
    if sub_xmls:
        return ET.parse(sub_xmls[0]).getroot(), sub_xmls[0]

    cvat_zips = glob.glob(os.path.join(category_dir, '*CVAT*.zip'))
    if cvat_zips:
        with zipfile.ZipFile(cvat_zips[0], 'r') as zf:
            with zf.open('annotations.xml') as f:
                return ET.parse(f).getroot(), f"{cvat_zips[0]} -> annotations.xml"

    raise FileNotFoundError(
        f"Δεν βρέθηκε `annotations.xml` ή `*CVAT*.zip` μέσα στον φάκελο: {category_dir}"
    )


def find_images_subdir(category_dir, masks_dir_name='masks'):
    """
    Εντοπίζει αυτόματα τον υποφάκελο που περιέχει τις εικόνες υπερήχων (.png)
    μέσα στον φάκελο της κατηγορίας (π.χ. `Orginal_train_images_to_959_661`, `Trans-cerebellum` κ.λπ.).
    """
    valid_exts = ('.png', '.jpg', '.jpeg')
    best_dir = None
    best_count = 0

    for entry in sorted(os.listdir(category_dir)):
        full_path = os.path.join(category_dir, entry)
        if not os.path.isdir(full_path):
            continue
        if entry == masks_dir_name or 'cvat' in entry.lower() or 'segmentation' in entry.lower():
            continue

        img_count = sum(
            1 for f in os.listdir(full_path)
            if f.lower().endswith(valid_exts)
        )
        if img_count > best_count:
            best_count = img_count
            best_dir = full_path

    return best_dir, best_count


def process_category(category_dir, masks_subdir='masks'):
    """
    Διαβάζει το `annotations.xml` μιας κατηγορίας και παράγει όλες τις μάσκες στον υποφάκελο `masks/`.
    """
    xml_root, xml_source = find_annotations_xml(category_dir)
    images_dir, disk_img_count = find_images_subdir(category_dir, masks_dir_name=masks_subdir)

    out_masks_dir = os.path.join(category_dir, masks_subdir)
    os.makedirs(out_masks_dir, exist_ok=True)

    stats = {
        'images_in_xml': 0,
        'images_on_disk': disk_img_count,
        'masks_saved': 0,
        'Brain': 0,
        'CSP': 0,
        'LV': 0,
        'verified_non_empty_CSP': 0,
        'verified_non_empty_LV': 0,
    }

    for img_elem in xml_root.findall('image'):
        stats['images_in_xml'] += 1
        img_name = img_elem.attrib['name']
        width = int(img_elem.attrib.get('width', 959))
        height = int(img_elem.attrib.get('height', 661))

        mask_rgb = np.zeros((height, width, 3), dtype=np.uint8)

        # Συλλέγουμε όλα τα σχήματα της εικόνας και τα ταξινομούμε με βάση το DRAW_PRIORITY
        # ώστε το Brain (1) να ζωγραφίζεται ΠΑΝΤΑ πρώτο και τα CSP (2) / LV (3) από πάνω του!
        shapes = []
        for child in img_elem:
            if child.tag in ('ellipse', 'box'):
                label = child.attrib.get('label', '')
                if label in CLASS_COLORS:
                    priority = DRAW_PRIORITY.get(label, 99)
                    shapes.append((priority, child.tag, label, child.attrib))

        shapes.sort(key=lambda item: item[0])

        has_csp = False
        has_lv = False

        for _, tag, label, attrib in shapes:
            color = CLASS_COLORS[label]
            rot = float(attrib.get('rotation', 0.0))

            if tag == 'ellipse':
                cx = float(attrib['cx'])
                cy = float(attrib['cy'])
                rx = float(attrib['rx'])
                ry = float(attrib['ry'])
                draw_rotated_ellipse(mask_rgb, cx, cy, rx, ry, rot, color)
            elif tag == 'box':
                xtl = float(attrib['xtl'])
                ytl = float(attrib['ytl'])
                xbr = float(attrib['xbr'])
                ybr = float(attrib['ybr'])
                draw_rotated_box(mask_rgb, xtl, ytl, xbr, ybr, rot, color)

            stats[label] += 1
            if label == 'CSP':
                has_csp = True
            elif label == 'LV':
                has_lv = True

        # Επιβεβαίωση ότι κανένα CSP ή LV δεν χάθηκε/σβήστηκε πάνω στη μάσκα
        if has_csp:
            csp_pixels = np.all(mask_rgb == CLASS_COLORS['CSP'], axis=-1)
            if np.any(csp_pixels):
                stats['verified_non_empty_CSP'] += 1
        if has_lv:
            lv_pixels = np.all(mask_rgb == CLASS_COLORS['LV'], axis=-1)
            if np.any(lv_pixels):
                stats['verified_non_empty_LV'] += 1

        out_path = os.path.join(out_masks_dir, img_name)
        Image.fromarray(mask_rgb, mode='RGB').save(out_path)
        stats['masks_saved'] += 1

    return stats, xml_source, images_dir, out_masks_dir


def discover_category_dirs(dataset_dir):
    """
    Βρίσκει τους 4 φακέλους κατηγοριών μέσα στο `dataset_dir`
    (υποστηρίζει τόσο την ονομασία `1`, `2`, `3`, `4` στον server όσο και
    την πλήρη ονομασία `1) Diverse...`, `2) Trans-cerebellum` κ.λπ.).
    """
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Ο φάκελος βάσης δεν βρέθηκε: {dataset_dir}")

    subdirs = []
    for entry in sorted(os.listdir(dataset_dir)):
        full_path = os.path.join(dataset_dir, entry)
        if os.path.isdir(full_path) and entry[0] in ('1', '2', '3', '4'):
            subdirs.append(full_path)

    if len(subdirs) != 4:
        raise RuntimeError(
            f"Αναμένονταν 4 φάκελοι κατηγοριών (1..4) μέσα στο {dataset_dir}, "
            f"αλλά βρέθηκαν {len(subdirs)}: {[os.path.basename(d) for d in subdirs]}"
        )
    return subdirs


def main():
    parser = argparse.ArgumentParser(
        description='Παραγωγή καθαρών μασκών Segmentation από τα CVAT annotations.xml'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default=os.path.expanduser('~/Dataset'),
        help='Μονοπάτι του φακέλου Dataset (προεπιλογή: ~/Dataset)'
    )
    parser.add_argument(
        '--masks-subdir',
        type=str,
        default='masks',
        help='Όνομα υποφακέλου όπου θα αποθηκευτούν οι μάσκες σε κάθε κατηγορία (προεπιλογή: masks)'
    )
    args = parser.parse_args()

    print("=" * 75)
    print(f"Έναρξη δημιουργίας καθαρών μασκών από CVAT XML")
    print(f"Φάκελος Dataset: {os.path.abspath(args.dataset_dir)}")
    print("=" * 75)

    category_dirs = discover_category_dirs(args.dataset_dir)

    totals = {
        'images_in_xml': 0,
        'images_on_disk': 0,
        'masks_saved': 0,
        'Brain': 0,
        'CSP': 0,
        'LV': 0,
        'verified_non_empty_CSP': 0,
        'verified_non_empty_LV': 0,
    }

    for cat_dir in category_dirs:
        cat_name = os.path.basename(cat_dir)
        print(f"\n[Κατηγορία {cat_name}] Επεξεργασία...")
        stats, xml_src, img_dir, out_dir = process_category(cat_dir, masks_subdir=args.masks_subdir)

        print(f"  Πηγή XML       : {xml_src}")
        print(f"  Φάκελος Εικόνων: {img_dir} ({stats['images_on_disk']} εικόνες)")
        print(f"  Φάκελος Μασκών : {out_dir} ({stats['masks_saved']} μάσκες αποθηκεύτηκαν)")
        print(
            f"  Δομές (Instances) -> Brain: {stats['Brain']} | "
            f"CSP: {stats['CSP']} (Επιβεβαιωμένες στη μάσκα: {stats['verified_non_empty_CSP']}) | "
            f"LV: {stats['LV']} (Επιβεβαιωμένες στη μάσκα: {stats['verified_non_empty_LV']})"
        )

        for k in totals:
            totals[k] += stats[k]

    print("\n" + "=" * 75)
    print("ΣΥΓΚΕΝΤΡΩΤΙΚΟΣ ΕΛΕΓΧΟΣ ΟΛΟΚΛΗΡΗΣ ΤΗΣ ΒΑΣΗΣ")
    print("=" * 75)
    print(f"  Σύνολο εικόνων στον δίσκο : {totals['images_on_disk']}")
    print(f"  Σύνολο μασκών που φτιάχτηκαν: {totals['masks_saved']}")
    print(f"  Σύνολο Brain instances    : {totals['Brain']} (Αναμενόμενα στο άρθρο: 3792)")
    print(
        f"  Σύνολο CSP instances      : {totals['CSP']} "
        f"(Ενεργά στη μάσκα: {totals['verified_non_empty_CSP']} / {totals['CSP']})"
    )
    print(
        f"  Σύνολο LV instances       : {totals['LV']} "
        f"(Ενεργά στη μάσκα: {totals['verified_non_empty_LV']} / {totals['LV']})"
    )
    lost_csp = totals['CSP'] - totals['verified_non_empty_CSP']
    lost_lv = totals['LV'] - totals['verified_non_empty_LV']
    print(f"  Χαμένες μάσκες λόγω επικάλυψης: CSP={lost_csp}, LV={lost_lv}")
    print("=" * 75)


if __name__ == '__main__':
    main()
