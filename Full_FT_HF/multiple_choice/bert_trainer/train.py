import evaluate
import numpy as np
from data_loader import swag, DataCollatorForMultipleChoice, preprocess_function
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer

device = "cuda:0"
model_name = "bert-base-uncased" 

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
preprocess = lambda x: preprocess_function(x, tokenizer)
tokenized_swag = swag.map(preprocess, batched=True)

model = AutoModelForMultipleChoice.from_pretrained(model_name)
model.to(device)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir=f"{model_name}-swag-finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_swag["train"],
    eval_dataset=tokenized_swag["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()