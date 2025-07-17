# Qwen1.5 Intent Classification Finetuning

This repository contains scripts to finetune Qwen1.5 models for dialogue intent classification using the DSTC8 schema-guided dialogue dataset.

## Files

- `qwen_intent_finetune.py` - Main finetuning script
- `inference.py` - Inference script for testing the finetuned model
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the Qwen1.5 model downloaded locally (e.g., in `./qwen_model_7B`)

3. Ensure your datasets are in the following directories:
   - `intent_classification_finetune_dataset/` - Training data
   - `intent_classification_dev_dataset/` - Validation data  
   - `intent_classification_test_dataset/` - Test data

## Dataset Format

Each file in the dataset should be a JSON object with the following structure:
```json
{
  "dialogue": "<USER> I'd like to find a hotel room.<SYSTEM> Which city are you searching in?<USER> London, UK please.<SYSTEM> I've found a 5 star hotel called 45 Park Lane.",
  "act": "NOTIFY_SUCCESS",
  "dialogue_id": "42_00061"
}
```

## Training

### Basic Training
```bash
python qwen_intent_finetune.py
```

### Advanced Training with Custom Parameters
```bash
python qwen_intent_finetune.py \
    --output_dir ./my_finetuned_model \
    --num_epochs 5 \
    --batch_size 1 \
    --learning_rate 3e-5 \
    --fp16 \
    --gradient_checkpointing
```

### Training Parameters

- `--output_dir`: Output directory for the finetuned model (default: `./qwen_intent_finetuned`)
- `--num_epochs`: Number of training epochs (default: 3)
- `--batch_size`: Training batch size (default: 2)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--warmup_steps`: Number of warmup steps (default: 100)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 4)
- `--save_steps`: Save checkpoint every X steps (default: 500)
- `--eval_steps`: Evaluate every X steps (default: 500)
- `--logging_steps`: Log every X steps (default: 100)
- `--fp16`: Use fp16 precision (default: False)
- `--gradient_checkpointing`: Use gradient checkpointing to save memory (default: False)

## Inference

### Basic Inference
```bash
python inference.py
```

### Inference with Custom Model Path
```bash
python inference.py --model_path ./my_finetuned_model
```

### Inference with Custom Dialogue
```bash
python inference.py --dialogue "<USER> I need a hotel in Paris.<SYSTEM> What's your budget?<USER> Around $200 per night.<SYSTEM> I found Hotel de la Paix, a 3-star hotel for $180 per night."
```

## Model Architecture

The script uses a causal language modeling approach where:
1. Dialogues are formatted into prompts with the target intent
2. The model learns to predict the intent by completing the prompt
3. During inference, the model generates the intent based on the dialogue context

## Memory Optimization

For large models, consider using:
- `--fp16`: Enable mixed precision training
- `--gradient_checkpointing`: Enable gradient checkpointing
- Reduce `--batch_size` if you run out of memory
- Increase `--gradient_accumulation_steps` to maintain effective batch size

## Example Usage

1. **Start Training**:
```bash
python qwen_intent_finetune.py --fp16 --gradient_checkpointing
```

2. **Monitor Training**: The script will show training progress and save checkpoints

3. **Test the Model**:
```bash
python inference.py --model_path ./qwen_intent_finetuned
```

4. **Interactive Testing**: The inference script provides an interactive mode where you can input dialogues and see predictions

## Expected Intents

Based on the DSTC8 dataset, the model will learn to classify intents such as:
- `INFORM_INTENT`
- `REQUEST`
- `CONFIRM`
- `OFFER_INTENT`
- `NOTIFY_SUCCESS`
- `GOODBYE`
- And many others...

## Troubleshooting

1. **Out of Memory**: Reduce batch size or enable gradient checkpointing
2. **Slow Training**: Enable fp16 precision or reduce sequence length
3. **Poor Results**: Try adjusting learning rate or increasing training epochs
4. **Model Loading Issues**: Ensure the model path is correct and the model files are complete

## Performance Tips

- Use a GPU with sufficient VRAM (16GB+ recommended for 7B model)
- Enable mixed precision training with `--fp16`
- Use gradient checkpointing for memory efficiency
- Monitor training loss to avoid overfitting
- Use early stopping to prevent overfitting 