"""
This script is adapted from:
https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/paraphrases/training.py
"""

import sys
sys.path.append('.')
import os
import argparse
import gzip
import json
import logging
import math
import sys

from sentence_transformers import models, losses, datasets, LoggingHandler, SentenceTransformer, InputExample, util

from transformers.adapters import AdapterConfig

from MultiDatasetDataLoader import MultiDatasetDataLoader


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


# Parse arguments
parser = argparse.ArgumentParser(description="Sentece embedding pre-training on general-domain sentence pairs or triplets")

parser.add_argument(
    "--model_name_or_path",
    type=str,
    default=None,
    help="The name or path of the underlying Transformer model to use.",
)

parser.add_argument(
    "--use_adapter",
    type=bool,
    default=False,
    help="Whether to train SEPT on an adapter or the full Transformer model.",
)

parser.add_argument(
    "--adapter_config",
    type=str,
    default='parallel',
    help="Adapter config (via AdapterHub).",
)

parser.add_argument(
    "--adapter_name",
    type=str,
    default='sept',
    help="Name of the trained adapter.",
)

parser.add_argument(
    "--batch_size_pairs",
    type=int,
    default=64,
    help="Batch size for sentence pair data.",
)

parser.add_argument(
    "--batch_size_triplets",
    type=int,
    default=64,
    help="Batch size for sentence triplets data.",
)

parser.add_argument(
    "--max_seq_length",
    type=int,
    default=512,
    help="Max sequence length of the Transformer.",
)

parser.add_argument(
    "--pooling_mode",
    type=str,
    default='mean',
    help="Pooling method for creating sentence embedding.",
)

parser.add_argument(
    "--learning_rate",
    type=float,
    default=1e-4,
    help="Learning rate for sentence-embedding training.",
)

parser.add_argument(
    "--num_epochs",
    type=int,
    default=1,
    help="Number of training epochs.",
)

parser.add_argument(
    "--use_amp",
    type=bool,
    default=False,
    help="Set to False, if you use a CPU or your GPU does not support FP16 operations.",
)

parser.add_argument(
    "--model_save_path",
    type=str,
    default=None,
    help="Save path of the trained Transformer model or adapter.",
)

args = parser.parse_args()

model_name_or_path = args.model_name_or_path
use_adapter = args.use_adapter
adapter_config = args.adapter_config
adapter_name = args.adapter_name
model_save_path = args.model_save_path
num_epochs = args.num_epochs
batch_size_pairs = args.batch_size_pairs
batch_size_triplets = args.batch_size_pairs
max_seq_length = args.max_seq_length
pooling = args.pooling_mode
lr = args.learning_rate
use_amp = args.use_amp


# Download SEPT data if necessary
dataset_paths = [
    'data/SEPT/AllNLI.jsonl.gz',
    'data/SEPT/sentence-compression.jsonl.gz',
    'data/SEPT/stackexchange_duplicate_questions.jsonl.gz'
]
for dataset_path in dataset_paths:
    if not os.path.exists(dataset_path):
        file_name = dataset_path.split('/')[-1]
        util.http_get(f'https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/paraphrases/{file_name}', dataset_path)

# Create SentenceTransformer model
word_embedding_model = models.Transformer(model_name_or_path, max_seq_length=max_seq_length)

if use_adapter:
    logging.info("Adding adapter...")
    # Add and activate adapter
    adapter_config = AdapterConfig.load(adapter_config)
    word_embedding_model.auto_model.add_adapter(adapter_name, config=adapter_config)
    word_embedding_model.auto_model.set_active_adapters(adapter_name)
    word_embedding_model.auto_model.train_adapter(adapter_name)
else:
    logging.info("SEPT without adapter")

pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=pooling)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Create training dataloader
datasets = []
for filepath in dataset_paths:
    dataset = []
    with gzip.open(filepath, 'rt', encoding='utf8') as fIn:
        for line in fIn:
            splits = json.loads(line)
            if type(splits) == dict:
                guid = splits['guid']
                texts = splits['texts']
            else:
                guid = None
                texts = splits
            dataset.append(InputExample(texts=texts, guid=guid))
    datasets.append(dataset)

train_dataloader = MultiDatasetDataLoader(datasets, 
                                          batch_size_pairs=batch_size_pairs, 
                                          batch_size_triplets=batch_size_triplets)

# Our training loss
train_loss = losses.MultipleNegativesRankingLoss(model)

# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up 
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          use_amp=use_amp,
          optimizer_params={'lr': lr},
        )

if use_adapter:
    # Save SEPT-ed adapter
    model._first_module().auto_model.save_adapter(model_save_path, adapter_name)
else:
    # Save SEPT-ed Transformer
    model.save(model_save_path)
