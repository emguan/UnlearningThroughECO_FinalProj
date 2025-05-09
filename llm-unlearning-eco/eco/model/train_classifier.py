import os
import json
from datasets import Dataset, load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Load forget and retain subsets from Hugging Face
forget_ds = load_dataset("locuslab/TOFU", name="forget01", split="train")
retain_ds = load_dataset("locuslab/TOFU", name="retain99", split="train")

# Label and merge
data = [{"text": item["prompt"], "label": 1} for item in forget_ds] + \
       [{"text": item["prompt"], "label": 0} for item in retain_ds]
dataset = Dataset.from_list(data)

# Tokenize
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

tokenized_dataset = dataset.map(tokenize, batched=True)

# Load model
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir="tofu_classifiers/forget01",
    evaluation_strategy="no",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=2e-5,
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

trainer.train()
