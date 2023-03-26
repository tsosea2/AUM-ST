#!/bin/bash

AUGDIR=
INPUTDATASET=

for VARIABLE in 2 3 4 5 6 7 8 9 10
do
python augment.py --augmentation_dir $AUGDIR --num_replicated_augmentations 50 --backtranslation --chain_length $VARIABLE --synonym_replacement --num_synonym_replacement 0 --random_deletion --prob_random_deletion 0 --dataset_path $INPUTDATASET
done

for VARIABLE in 2 3 4 5 6 7 8 9 10
do
python augment.py --augmentation_dir $AUGDIR --num_replicated_augmentations 50 --backtranslation --chain_length $VARIABLE --synonym_replacement --num_synonym_replacement 1 --random_deletion --prob_random_deletion 0 --dataset_path $INPUTDATASET
done

for VARIABLE in 2 3 4 5 6 7 8 9 10
do
python augment.py --augmentation_dir $AUGDIR --num_replicated_augmentations 50 --backtranslation --chain_length $VARIABLE --synonym_replacement --num_synonym_replacement 1 --random_deletion --prob_random_deletion 0.05 --dataset_path $INPUTDATASET
done

for VARIABLE in 2 3 4 5 6 7 8 9 10
do
python augment.py --augmentation_dir $AUGDIR --num_replicated_augmentations 50 --backtranslation --chain_length $VARIABLE --synonym_replacement --num_synonym_replacement 0 --random_deletion --prob_random_deletion 0.05 --dataset_path $INPUTDATASET
done