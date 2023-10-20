import argparse
import logging

import pandas as pd

from datasets import concatenate_datasets, Value, Dataset

from setfit import SetFitModel, SetFitTrainer

from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier

from sentence_transformers import SentenceTransformer, models, LoggingHandler


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


parser = argparse.ArgumentParser(description="SetFit with self-training.")

parser.add_argument(
    "--model_name_or_path",
    type=str,
    default=None,
    help="The name or path of underlying Transformer model to use.",
)

parser.add_argument(
    "--adapter_path",
    type=str,
    default=None,
    help="Path of the SEPT adapter.",
)

parser.add_argument(
    "--adapter_name",
    type=str,
    default=None,
    help="Name of the SEPT adapter.",
)

parser.add_argument(
    '--dataset_id', 
    type=str,
    default=None,
    help="The name of the dataset to use (via the datasets library)."
)

parser.add_argument(
    "--unlabeled_file_path",
    type=str,
    default=None,
    required=True,
    help="Path of a local .txt file containing unlabeled data, one sample per line.",
)

parser.add_argument(
    "--train_dataset_path",
    type=str,
    default=None,
    required=True,
    help="Path of a local .csv file containing labeled few-shot training data.",
)

parser.add_argument(
    "--eval_dataset_path",
    type=str,
    default=None,
    required=True,
    help="Path of a local .csv file containing evaluation data.",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=16,
    help="Batch size for contrastive fine-tuning.",
)

parser.add_argument(
    "--num_epochs",
    type=int,
    default=1,
    help="Number of training epochs.",
)

parser.add_argument(
    "--num_iterations",
    type=int,
    default=20,
    help="The number of text pairs to generate for contrastive learning for each sentence.",
)

parser.add_argument(
    "--num_samples",
    type=int,
    default=8,
    help="Number of labeled samples per class to use.",
)

parser.add_argument(
    "--text_col",
    type=str,
    default='text',
    help="Name of the text column in the dataset.",
)

parser.add_argument(
    "--label_col",
    type=str,
    default='label',
    help="Name of the label column in the dataset.",
)

parser.add_argument(
    "--model_save_path",
    type=str,
    default=None,
    help="Save path of the model",
)

parser.add_argument(
    "--threshold",
    type=float,
    default=0.9,
    help="Pseudo-labels with prediction probabilities above threshold are added to the training dataset",
)


args = parser.parse_args()

num_samples = args.num_samples
num_iterations = args.num_iterations
num_epochs = args.num_epochs
batch_size = args.batch_size
threshold = args.threshold

model_path = args.model_name_or_path
adapter_path = args.adapter_path
adapter_name = args.adapter_name
model_save_path = args.model_save_path
unlabeled_file_path = args.unlabeled_file_path

text_col = args.text_col
label_col = args.label_col

# Load datasets
df_train = pd.read_csv(args.train_dataset_path)
df_eval = pd.read_csv(args.eval_dataset_path)
train_dataset = Dataset.from_pandas(df_train)
eval_dataset = Dataset.from_pandas(df_eval)

# Create Sentence Transformer
if adapter_path:
    word_embedding_model = models.Transformer(model_path)
    # Load and activate adapter
    word_embedding_model.auto_model.load_adapter(args.adapter_path)
    word_embedding_model.auto_model.set_active_adapters(args.adapter_name)
    # Turn on gradient update for adapter parameters 
    word_embedding_model.auto_model.train_adapter(args.adapter_name)
    # Turn on gradient update for Transformer parameters 
    word_embedding_model.auto_model.freeze_model(False)
    
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
else: 
    model = SentenceTransformer(model_path)


# SetFit 
setfit_model = SetFitModel(model_body=model,
                           model_head=LogisticRegression())

# Train a setfit body with few-shot labeled sentences     
trainer = SetFitTrainer(
    model=setfit_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    batch_size=batch_size,
    num_epochs=num_epochs,
    num_iterations=num_iterations,
    column_mapping={text_col: "text", label_col: "label"},
)
trainer.train()

metrics = trainer.evaluate()
print("Result without self-training: ", metrics)


# Self-train on a Logistic Regression head with both labeled and unlabeled data
logging.info("Start self-training...")

# Create new dataset with golden few-shot data + unlabeled data 
unlabeled_texts = []
with open (unlabeled_file_path) as f:
    for line in f:
        line = line.strip()
        if line not in train_dataset[text_col]:
            unlabeled_texts.append(line)

unlabeled_train_dataset = Dataset.from_dict({
    text_col: unlabeled_texts,
    label_col: [-1] * len(unlabeled_texts) 
})

train_dataset = train_dataset.cast_column('label', Value("int64"))
self_training_dataset = concatenate_datasets([train_dataset, unlabeled_train_dataset]).shuffle()

# Use the setfit model body trained with few-shot golden data to encode self-training data
encoded_x_train = setfit_model.model_body.encode(self_training_dataset['text'])

# Train the self-training classifier
self_training_model = SelfTrainingClassifier(LogisticRegression(), threshold=threshold)
self_training_model.fit(encoded_x_train, self_training_dataset['label'])

# Evaluate on test set
x_test = eval_dataset["text"]
acc = self_training_model.score(setfit_model.model_body.encode(x_test), eval_dataset["label"])

print("Result with self-training:", acc)
