# AUM-ST

This is the code for our EMNLP Findings 2022 paper [Leveraging Training Dynamics and Self-Training for Text Classification]([https://www.aclweb.org/anthology/2020.emnlp-main.715/](https://aclanthology.org/2022.findings-emnlp.350/)). If you use this repository in your research, please cite our paper:

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
}
```
## Abstract

The effectiveness of pre-trained language models in downstream tasks is highly dependent on the amount of labeled data available for training. Semi-supervised learning (SSL) is a promising technique that has seen wide attention recently due to its effectiveness in improving deep learning models when training data is scarce. Common approaches employ a teacher-student self-training framework, where a teacher network generates pseudo-labels for unlabeled data, which are then used to iteratively train a student network. In this paper, we propose a new self-training approach for text classification that leverages training dynamics of unlabeled data. We evaluate our approach on a wide range of text classification tasks, including emotion detection, sentiment analysis, question classification and gramaticality, which span a variety of domains, e.g, Reddit, Twitter, and online forums. Notably, our method is successful on all benchmarks, obtaining an average increase in F1 score of 3.5{\%} over strong baselines in low resource settings.

## Overview
