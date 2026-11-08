#!/bin/bash
#SBATCH --job-name=unet_train
#SBATCH --gres=gpu:1
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log

# Μετάβαση στον φάκελο από όπου υποβλήθηκε η εργασία
cd "$SLURM_SUBMIT_DIR"

# Ενεργοποίηση του Python περιβάλλοντος
source ~/unet_env/bin/activate

# Εκτέλεση του train.py περνώντας όλα τα ορίσματα του sbatch (-u = ζωντανή καταγραφή στο log)
python -u train.py "$@"
