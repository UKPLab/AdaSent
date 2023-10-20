import argparse
import logging
from datasets import Dataset
import pandas as pd
from setfit import SetFitModel, SetFitTrainer
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer, models, LoggingHandler


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


parser = argparse.ArgumentParser(description="Sentence embedding fine-tuning (SetFit) for few-shot classification")

parser.add_argument(
    "--model_name_or_path",
    type=str,
    default=None,
    required=True,
    help="The name or path of DAPT-ed Transformer model to use.",
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
    "--train_dataset_path",
    type=str,
    default=None,
    required=True,
    help="Path of a local train .csv file.",
)

parser.add_argument(
    "--eval_dataset_path",
    type=str,
    default=None,
    required=True,
    help="Path of a local eval .csv file",
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
    help="The number of text pairs to generate for each sentence for the contrastive learning.",
)

parser.add_argument(
    "--num_samples",
    type=int,
    default=8,
    help="Number of labeled samples per class.",
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

args = parser.parse_args()

num_samples = args.num_samples
num_iterations = args.num_iterations
num_epochs = args.num_epochs
batch_size = args.batch_size

model_path = args.model_name_or_path
adapter_path = args.adapter_path
adapter_name = args.adapter_name
model_save_path = args.model_save_path

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

print(metrics)

if model_save_path:
    setfit_model._save_pretrained(model_save_path)
