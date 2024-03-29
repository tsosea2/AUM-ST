# AUM-ST

This is the code for our Findings of EMNLP 2022 paper entitled [Leveraging Training Dynamics and Self-Training for Text Classification]([https://www.aclweb.org/anthology/2020.emnlp-main.715/](https://aclanthology.org/2022.findings-emnlp.350/)). If you use this repository in your research, please cite our paper:

```bibtex
@inproceedings{sosea-caragea-2022-leveraging,
    title = "Leveraging Training Dynamics and Self-Training for Text Classification",
    author = "Sosea, Tiberiu  and
      Caragea, Cornelia",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.350",
    pages = "4750--4762",
}
```
## Abstract

The effectiveness of pre-trained language models in downstream tasks is highly dependent on the amount of labeled data available for training. Semi-supervised learning (SSL) is a promising technique that has seen wide attention recently due to its effectiveness in improving deep learning models when training data is scarce. Common approaches employ a teacher-student self-training framework, where a teacher network generates pseudo-labels for unlabeled data, which are then used to iteratively train a student network. In this paper, we propose a new self-training approach for text classification that leverages training dynamics of unlabeled data. We evaluate our approach on a wide range of text classification tasks, including emotion detection, sentiment analysis, question classification and gramaticality, which span a variety of domains, e.g, Reddit, Twitter, and online forums. Notably, our method is successful on all benchmarks, obtaining an average increase in F1 score of 3.5{\%} over strong baselines in low resource settings.

## Overview

AUM-ST can be divided into two main steps: ```Augmentation``` and ```SSL Training```. Due to high computational costs of augmentations such as Backtranslation, we resort to generating the needed augmentations offline, before starting to train. As mentioned in the paper, we use transformations such as synonym replacement, switchout, and backtranslations with various chain lengths.

## Dependencies

To install all dependencies please run:

```pip install -r requirements.txt```

## Data acquisition

Unfortunately, we cannot upload the datasets used in the paper. However these can easily be downloaded from their sources and put in the correct format.

SST-2, SST-5 -> [https://nlp.stanford.edu/sentiment/](https://nlp.stanford.edu/sentiment/)

CancerEmo -> [https://github.com/tsosea2/CancerEmo](https://github.com/tsosea2/CancerEmo)

GoEmotions -> [https://github.com/google-research/google-research/tree/master/goemotions](https://github.com/google-research/google-research/tree/master/goemotions)

TREC-6 -> [https://www.kaggle.com/datasets/thedevastator/the-trec-question-classification-dataset-a-longi](https://www.kaggle.com/datasets/thedevastator/the-trec-question-classification-dataset-a-longi)

COLA -> [https://nyu-mll.github.io/CoLA/](https://nyu-mll.github.io/CoLA/)

IMDB -> [https://ai.stanford.edu/~amaas/data/sentiment/](https://ai.stanford.edu/~amaas/data/sentiment/)


## Augmentation

### Data Format

The input data for the augmentation stage is a csv file containing the training set with three columns: 

```Id``` - unique ID

```Text``` - the input text

```Label``` - the label of the example as an integer. For instance, for a dataset with `N` classes, the Label has to be a number from $0$ to $N-1$.

### Augmenting the training set.

To replicate the exact augmentations in the paper, first fill out the `AUGDIR` and `INPUTDATASET` variables in `Augment/generate_augmentations.sh`. `AUGDIR` is the directory where the augmentations will be saved while `INPUTDATASET` is the path to `.csv` file defined above. Next, generate the augmentations by running

`sh Augment/generate_augmentations.sh`

While augmentation is a one-time process, please note that backtranslations are computationally expensive. Therefore, if multiple GPUs are available, consider running the strong augmentations in parallel on multiple GPUs.

### Output of the augmentation

The augmentation script generates a large amount of augmented versions of the training set, replicated multiple times, in order to mimic the stochastic process usually used in augmentations. For each configuration defined in `Augment/generate_augmentations.sh` we replicate the augmentations for a number of $50$ times through the `--num_replicated_augmentations` flag. To save time and resources, this number can be lowered, however, please note that the results in the paper use $50$ replicas.

Each augmentation in `AUGDIR` will follow the same `.csv` format described above. In addition, the script adds a extra column named `Strength` which indicates the amount of distorsion produced by the augmentation (i.e., higher strengths correspond to more backtranslation chain lengths while lower strenghts mean fewer chains). These values can be used in AUM-ST to define weak and strong augmentations.

### Strength explanation

This paper is based on three main augmentations: synonym replacement, switchout and backtranslation. Each augmentation has a strength associated with it. Each synonym replacement has a strength of $1$, deletion has a strength of $1$ for each $0.05$ probability, while each backtranslation intermediate language has a strength of $3$. For instance, if we perform $1$ synonym replacement, set a switchout probability of $0.1$ and use $3$ intermediate languages for backtranslation (i.e., chain length of $4$), the resulted augmentation has a strength of $1$(synonym replacement) + $2$ (switchout) + $3\*3$ (backtranslation) = $12$. These strength values can be used to define what weak or strong data augmnetation represents in AUM-ST. 

## AUM-ST

The previous section discussed the augmentation process. At this stage, we assume we have the augmentations generated in the `AUGDIR` directory. We also assume the validation and the test set follow the same format as the training set (i.e., `.csv` file with `Id`, `Text`, and `Label` columns). To train using AUM-ST run:

```python AUM-ST/aum-st.py --experiment_id <experiment_id> --intermediate_model_path <intermediate_model_path> --num_labels <num_labels> --augmentation_dir AUGDIR --validation_path <path_to_validation.csv> --test_path <path_to_test_set.csv> --tensorboard_dir <path_to_tensorboard> --aum_save_dir <path_to_aum_savedir>```

There exist numerous other flags that are set to the default value used in the paper. If you want to change any parameters, please see in-depth details of each flag by running:

```python AUM-ST/aum-st.py --helpfull```

## Monitoring

All metrics are logged during training using tensorboard. To monitor these, please run 

```tensorboard --logdir .```

## Help

If you have any questions or need help running the code, please create an issue in this repository.


