#!/usr/bin/env python3
"""
Qwen1.5 Intent Classification Finetuning Script
Finetunes Qwen1.5 model on dialogue intent classification task
"""

import os
import json
import torch
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import Dataset
import numpy as np
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """Arguments for model configuration"""
    model_name_or_path: str = field(
        default="./qwen_model_1_5B",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading a model"}
    )
    torch_dtype: str = field(
        default="auto",
        metadata={"help": "Override the default `torch.dtype` and load the model under this dtype"}
    )

@dataclass
class DataTrainingArguments:
    """Arguments for data training configuration"""
    train_dataset_path: str = field(
        default="intent_classification_finetune_dataset",
        metadata={"help": "Path to the training dataset directory"}
    )
    val_dataset_path: str = field(
        default="intent_classification_dev_dataset",
        metadata={"help": "Path to the validation dataset directory"}
    )
    test_dataset_path: str = field(
        default="intent_classification_test_dataset",
        metadata={"help": "Path to the test dataset directory"}
    )
    max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )

def load_dataset(data_path: str) -> List[Dict]:
    """Load the intent classification dataset"""
    dataset = []
    
    for filename in os.listdir(data_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(data_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    line = f.readline().strip()
                    data = json.loads(line)
                    dataset.append(data)
            except Exception as e:
                logger.warning(f"Error loading {filename}: {e}")
                continue
    
    logger.info(f"Loaded {len(dataset)} samples from dataset")
    return dataset

def create_prompt_template(dialogue: str, act: str) -> str:
    """Create a prompt template for the model"""
    prompt = f"""Below is a dialogue between a user and a system. Please classify the intent/action of the system's response.

Dialogue:
{dialogue}

Intent/Action: {act}

### Response:
The intent/action is: {act}"""
    
    return prompt

def prepare_dataset(data: List[Dict], tokenizer, max_length: int) -> Dataset:
    """Prepare the dataset for training"""
    processed_data = []
    
    for item in data:
        dialogue = item['dialogue']
        act = item['act']
        
        # Create prompt
        prompt = create_prompt_template(dialogue, act)
        
        # Tokenize
        tokenized = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None
        )
        
        # Add labels (same as input_ids for causal language modeling)
        # tokenized['labels'] = tokenized['input_ids'].copy()
        
        processed_data.append(tokenized)
    
    return Dataset.from_list(processed_data)

def get_unique_acts(dataset: List[Dict]) -> List[str]:
    """Get unique intent/action labels"""
    acts = set()
    for item in dataset:
        acts.add(item['act'])
    return sorted(list(acts))

def main():
    parser = argparse.ArgumentParser(description="Finetune Qwen1.5 for intent classification")
    parser.add_argument("--output_dir", type=str, default="./qwen_intent_finetuned", 
                       help="Output directory for the finetuned model")
    parser.add_argument("--num_epochs", type=int, default=3, 
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, 
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                       help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, 
                       help="Number of warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, 
                       help="Gradient accumulation steps")
    parser.add_argument("--save_steps", type=int, default=500, 
                       help="Save checkpoint every X steps")
    parser.add_argument("--eval_steps", type=int, default=500, 
                       help="Evaluate every X steps")
    parser.add_argument("--logging_steps", type=int, default=100, 
                       help="Log every X steps")
    parser.add_argument("--fp16", action="store_true", 
                       help="Use fp16 precision")
    parser.add_argument("--gradient_checkpointing", action="store_true", 
                       help="Use gradient checkpointing to save memory")
    parser.add_argument("--max_steps", type=int, default=None,
                       help="Maximum number of training steps (overrides num_epochs)")
    
    args = parser.parse_args()
    
    # Load datasets
    logger.info("Loading training dataset...")
    train_data = load_dataset(DataTrainingArguments.train_dataset_path)
    
    logger.info("Loading validation dataset...")
    val_data = load_dataset(DataTrainingArguments.val_dataset_path)
    
    logger.info("Loading test dataset...")
    test_data = load_dataset(DataTrainingArguments.test_dataset_path)
    
    if not train_data:
        raise ValueError(f"No data found in the training dataset directory: {DataTrainingArguments.train_dataset_path}")
    if not val_data:
        raise ValueError(f"No data found in the validation dataset directory: {DataTrainingArguments.val_dataset_path}")
    if not test_data:
        raise ValueError(f"No data found in the test dataset directory: {DataTrainingArguments.test_dataset_path}")
    
    # Get unique acts from all datasets
    all_data = train_data + val_data + test_data
    unique_acts = get_unique_acts(all_data)
    logger.info(f"Found {len(unique_acts)} unique intent/action labels: {unique_acts}")
    
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Load tokenizer and model
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(
        ModelArguments.model_name_or_path,
        trust_remote_code=ModelArguments.trust_remote_code,
        padding_side="right"
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        ModelArguments.model_name_or_path,
        trust_remote_code=ModelArguments.trust_remote_code,
        torch_dtype=getattr(torch, ModelArguments.torch_dtype) if ModelArguments.torch_dtype != "auto" else "auto",
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Prepare datasets
    logger.info("Preparing datasets...")
    train_dataset = prepare_dataset(train_data, tokenizer, DataTrainingArguments.max_length)
    val_dataset = prepare_dataset(val_data, tokenizer, DataTrainingArguments.max_length)
    test_dataset = prepare_dataset(test_data, tokenizer, DataTrainingArguments.max_length)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal language modeling
        pad_to_multiple_of=8,  # For better performance on GPU
    )
    
    # Training arguments
    training_args_dict = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "remove_unused_columns": False,
        "dataloader_pin_memory": False,
        "report_to": None,  # Disable wandb/tensorboard logging
    }
    
    # Add num_train_epochs or max_steps, but not both
    if args.max_steps is not None:
        training_args_dict["max_steps"] = args.max_steps
    else:
        training_args_dict["num_train_epochs"] = args.num_epochs
    
    training_args = TrainingArguments(**training_args_dict)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    logger.info("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    logger.info(f"Test results: {test_results}")
    
    # Save test results
    with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)
    
    logger.info(f"Training completed! Model saved to {args.output_dir}")

if __name__ == "__main__":
    main() 