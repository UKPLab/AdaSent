# AdaSent: Efficient Domain-Adapted Sentence Embeddings for Few-Shot Classification

AdaSent is an approach to creating domain-specialized sentence encoders for few-shot sentence classification. It combines Domain-Adaptive Pre-training (DAPT) and Sentence Embedding Pre-Training (SEPT) in a modular fashion. First, it does DAPT on a base pre-trained language model (PLM). Separately, an adapter is trained through general-domain SEPT on the same PLM. The adapter stores the sentence-specialization abilities and can be plugged into domain-adapted PLMs from various domains to make them domain-specialized sentence encoders, on which [SetFit](https://github.com/huggingface/setfit) is carried out to do downstream few-shot classification training. :page_facing_up:[Paper](https://arxiv.org/abs/2311.00408)

<img src="https://github.com/UKPLab/AdaSent/assets/56653470/9b15da03-78ed-48a3-adf0-368305964204.png" width=35% height=35%>

## Project structure
```
├── data                <- Path to store data for training and inference when using this project
├── models              <- Path to store trained models when using this project
├── scripts             <- Scripts for training and inference
├── example.sh          <- Example script for AdaSent training
├── LICENSE  
├── NOTICE              <- Copyright information
├── README.md           <- The top-level README for developers using this project
└── requirements.txt    <- Requirements  
```

## Requirements
* Python 3.8

## Installation
* Install the requirements:

    ```
    pip install -r requirements.txt
    ```

## Train AdaSent
AdaSent consists of three part of training: DAPT, SEPT and SetFit. The following provides explanations for each of them. 

An example script including all three parts can be found [here](example.sh). The script should be run from the project root. 

### Data Preparation
 
To train AdaSent, you need to prepare
- A `.txt` file containing unlabeled examples (one example per line) for DAPT  
- A `.csv` file with labeled training data and a `.csv` file with evaluation data for the few-shot classification task, and fill in the paths in the script.   

Take the task [`mteb/amazon_massive_scenario`](https://huggingface.co/datasets/mteb/amazon_massive_scenario) for example, you can create these data files with the following code:
```python
import pandas as pd
from datasets import load_dataset
from setfit import sample_dataset

# Paths to store files
unlabeled_text_file = 'data/DAPT/amazon_massive_scenario.txt'
labeled_train_file = 'data/SetFit/amazon_massive_scenario_train.csv'
eval_file = 'data/SetFit/amazon_massive_scenario_eval.csv'

# Load dataset from Huggingface
dataset = load_dataset('mteb/amazon_massive_scenario', 'en')

# Write training data text in into a text file:
with open(unlabeled_text_file, 'w') as f:
    f.write('\n'.join(dataset['train']['text']))

# Sample few-shot labeled data out of the original training set
train_dataset = sample_dataset(dataset['train'], 
                               label_column='label', 
                               num_samples=8)
eval_dataset = dataset['test']

# Save train and evaluation data to .csv files
df = pd.DataFrame({'text': train_dataset['text'], 'label': train_dataset['label']})
df.to_csv(labeled_train_file)

df = pd.DataFrame({'text': eval_dataset['text'], 'label': eval_dataset['label']})
df.to_csv(eval_file)
``` 
 
### Domain-Adaptive Pre-Training (DAPT)
First, we need to train a domain-adapted PLM. The following command trains a `DistilRoBERTa` model on task-specific unlabeled data with MLM (The `--model_name_or_path` can be other local or Hugging Face model path):

```bash
python scripts/DAPT/train_mlm.py \
    --train_file data/DAPT/amazon_massive_scenario.txt \
    --model_name_or_path distilroberta-base \
    --max_seq_length 512 \
    --max_train_steps 2000 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --output_dir models/distilroberta-base_dapt_amazon_massive_scenario \
    --line_by_line True
```
This will save the DAPT-ed model at the specified `--output_dir`.

### Sentence Embedding Pre-Training (SEPT) on adapter
We train a SEPT adapter on the same base PLM (`DistilRoBERTa` in our case) as in DAPT. After training, this adapter can be inserted into any other domain-adapted PLM.  
In our work, we found that SEPT with the three datasets `AllNLI`, `Sentence compression` and `Stackexchange duplicate question` (see the datasets' information [here](https://www.sbert.net/examples/training/paraphrases/README.html)) can best improve downstream few-shot classification tasks. 

The following command trains an SEPT adapter on an unadapted `DistilRoBERTa` model and saves the adapter at `--model_save_path`. The above-mentioned datasets will be automatically downloaded. More information about the parameters can be found in the script [`train_sept`](scripts/SEPT/train_sept.py): 
```bash
python scripts/SEPT/train_sept.py \
    --model_name_or_path distilroberta-base \
    --use_adapter True \
    --adapter_config parallel \
    --adapter_name sept \
    --max_seq_length 512 \
    --batch_size_pairs 64 \
    --batch_size_triplets 64 \
    --num_epochs 1 \
    --learning_rate 1e-4 \
    --pooling_mode mean \
    --model_save_path models/distilroberta-base_sept_adapter \
    --use_amp True
```

### SetFit
After the DAPT-ed PLM and the SEPT-ed adapter are trained, we assemble them together and train a [SetFit](https://github.com/huggingface/setfit) model with 8 shots per class. Here, you need to specify the paths to the train and evalutaion `.csv` files in `--train_dataset_path` and `--eval_dataset_path` respectively 
:
```bash
python scripts/SetFit/train_setfit.py \
    --model_name_or_path models/distilroberta-base_dapt_amazon_massive_scenario \
    --adapter_path models/distilroberta-base_sept_adapter \
    --batch_size 16 \
    --num_epochs 1 \
    --num_samples 8 \
    --num_iterations 20 \
    --adapter_name sept \
    --model_save_path models/adasent_setfit_amazon_massive_scenario \
    --train_dataset_path data/SetFit/amazon_massive_scenario_train.csv \
    --eval_dataset_path data/SetFit/amazon_massive_scenario_eval.csv \
    --text_col text \
    --label_col label
```

The `--model_name_or_path` should be the path of the DAPT-ed PLM, and the `--adapter_path` should be the path of the SEPT-ed adapter. The trained SetFit model will be saved at the `--model_save_path`. More information about the parameters can be found in the script [`train_setfit`](scripts/SetFit/train_setfit.py).

#### Semi-supervised SetFit
The unlabeled data used in DAPT can also be used for self-training in SetFit and further improve the result. First, we use SetFit model body (DAPT-Transformer + SEPT-adapter in our case) trained on few-shot labeled data to encode both the labeled and unlabeled data. Then we run self-training with the encoded data with the [`SelfTrainingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.SelfTrainingClassifier.html) from scikit-learn.   
Run the following command to do self-training with the unlabeled data at `--unlabeled_file_path`, here we use the same file as in DAPT:
```bash
python scripts/SetFit/train_setfit_with_self_training.py \
    --model_name_or_path models/distilroberta-base_dapt_amazon_massive_scenario \
    --adapter_path models/distilroberta-base_sept_adapter \
    --adapter_name sept \
    --unlabeled_file_path data/DAPT/amazon_massive_scenario.txt \
    --batch_size 16 \
    --num_epochs 1 \
    --num_samples 8 \
    --num_iterations 20 \
    --train_dataset_path data/SetFit/amazon_massive_scenario_train.csv \
    --eval_dataset_path data/SetFit/amazon_massive_scenario_eval.csv \
    --text_col text \
    --label_col label
```

## Datasets and Models
### Evaluation data
| Dataset    | Link |
| -------- | ------- |
| MTEB Benchmark  | [MTEB](https://huggingface.co/mteb)    |
| ADE    | [ade_corpus_v2](https://huggingface.co/datasets/ade_corpus_v2) |
| RCT    | [armanc/pubmed-rct20k](https://huggingface.co/datasets/armanc/pubmed-rct20k) |
| FPB    | [financial_phrasebank](https://huggingface.co/datasets/financial_phrasebank) |
| TFNS   | [zeroshot/twitter-financial-news-sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) |
| TFNT   | [zeroshot/twitter-financial-news-topic](https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic) |
| LED    | [lex_glue](https://huggingface.co/datasets/lex_glue/viewer/ledgar) |

### Pre-trained SEPT Adapter
The pre-trained adapter for `distilroberta-base` is available [here](https://huggingface.co/yoh/distilroberta-base-sept-adapter) at Huggingface. 


## Citation
Please use the following citation:
```bibtex
@inproceedings{huang-etal-2023-adasent,
    title = "{A}da{S}ent: Efficient Domain-Adapted Sentence Embeddings for Few-Shot Classification",
    author = "Huang, Yongxin  and
      Wang, Kexin  and
      Dutta, Sourav  and
      Patel, Raj  and
      Glava{\v{s}}, Goran  and
      Gurevych, Iryna",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.208",
    pages = "3420--3434",
}
```

Contact person: Yongxin Huang, firstname.lastname@tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 
