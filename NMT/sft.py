from datasets import Dataset, DatasetDict
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
import os

prefix = "translate Udmurt to Russian: "

def preprocess(examples):
    inputs = [prefix + example for example in examples["udm"]]
    model_inputs = tokenizer(
        inputs, text_target=examples["ru"], max_length=128, truncation=True)

    return model_inputs

if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "udmurt"

    model = 'bigscience/mt0-small'
    output_dir = '...'
    b_size = 1

    model = AutoModelForSeq2SeqLM.from_pretrained(model)
    model.train()

    train_data = pd.read_csv("...")
    val_data = pd.read_csv("...")

    tokenizer = AutoTokenizer.from_pretrained(model)

    training_args = Seq2SeqTrainingArguments(
        remove_unused_columns=True,
        output_dir="mt0-udmurt",
        overwrite_output_dir=True,
        eval_strategy='steps',
        save_strategy='steps',
        warmup_steps=1000,
        eval_steps=10_000,
        save_steps=25_000,
        learning_rate=1e-3,
        per_device_train_batch_size=b_size,
        per_device_eval_batch_size=1,
        save_total_limit=20,
        optim="adamw_torch",
        report_to="wandb",
        lr_scheduler_type="linear",
        predict_with_generate=False,
        prediction_loss_only=True,
        log_level='info',
        max_steps=100_000,
        push_to_hub=False, 
    )

    data_files = {'train': Dataset.from_pandas(train_data),
                  'validation': Dataset.from_pandas(val_data)}
    
    all_dataset = DatasetDict(data_files)

    tokenized_dataset = all_dataset.map(lambda batch: preprocess(
            examples=batch),
        batched=True
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    model.save_pretrained(output_dir + '_l/')
