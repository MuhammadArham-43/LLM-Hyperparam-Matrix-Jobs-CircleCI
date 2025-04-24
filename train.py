import os
import sys
import argparse
import wandb
import time
import uuid
from transformers import (
    GPT2Tokenizer,
    GPT2ForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", choices=["linear", "cosine", "constant"])
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.99)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    return parser.parse_args()

def main(args):
    wandb_api_key = os.getenv("WANDB_API_KEY")
    assert wandb_api_key and wandb_api_key != "", "WANDB API key is required"
    try:
        wandb.login(key=wandb_api_key)
    except Exception as e:
        print("Error logging into wandb. Wrong API key.")
        print(e)
        sys.exit(1)

    timestamp = time.strftime("%Y%m%d-%H")
    print(timestamp)
    wandb.init(
        project="llm-hyperparam-tuning",
        group=timestamp
    )
    wandb.log({"hyperparameters": vars(args)})

    dataset = load_dataset("emotion")
    dataset = dataset["train"].train_test_split(test_size=0.3, seed=42)
    train_data = dataset["train"].select(range(140))
    test_data = dataset["test"].select(range(60))

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=args.max_length)

    train_tokenized = train_data.map(tokenize).rename_column("label", "labels")
    test_tokenized = test_data.map(tokenize).rename_column("label", "labels")
    train_tokenized.set_format("torch")
    test_tokenized.set_format("torch")

    model = GPT2ForSequenceClassification.from_pretrained("distilgpt2", num_labels=6)
    model.config.pad_token_id = model.config.eos_token_id

    # Step 5: Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        lr_scheduler_type=args.lr_scheduler_type,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        disable_tqdm=True
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        tokenizer=tokenizer
    )

    # Step 7: Train and Evaluate
    trainer.train()
    metrics = trainer.evaluate()
    wandb.log(metrics)
    print(f"Training Arguments: {args}")
    print(f"Training complete! Eval loss: {metrics.get('eval_loss')}")
    
    wandb.finish()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)