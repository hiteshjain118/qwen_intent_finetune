#!/usr/bin/env python3
"""
Evaluate the accuracy of the finetuned Qwen1.5 intent classification model
"""

import json
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse
from typing import List, Dict, Tuple

def load_test_dataset(test_dataset_path: str) -> List[Dict]:
    """Load the test dataset"""
    dataset = []
    
    for filename in os.listdir(test_dataset_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(test_dataset_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    line = f.readline().strip()
                    data = json.loads(line)
                    dataset.append(data)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
    
    print(f"Loaded {len(dataset)} test samples")
    return dataset

def create_prompt(dialogue: str) -> str:
    """Create a prompt for inference"""
    prompt = f"""Below is a dialogue between a user and a system. Please classify the intent/action of the system's response.

Dialogue:
{dialogue}

Intent/Action:"""
    
    return prompt

def predict_intent(model, tokenizer, dialogue: str, max_new_tokens: int = 50) -> str:
    """Predict the intent for a given dialogue"""
    prompt = create_prompt(dialogue)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    
    # Move to device
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Use greedy decoding for evaluation
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the predicted intent (everything after "Intent/Action:")
    if "Intent/Action:" in generated_text:
        intent_part = generated_text.split("Intent/Action:")[-1].strip()
        # Clean up any extra text
        intent_part = intent_part.split("\n")[0].strip()
        return intent_part
    
    return "Unknown"

def evaluate_model(model_path: str, test_dataset_path: str) -> Dict:
    """Evaluate the model on the test dataset"""
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype="auto"
    )
    
    # Load test dataset
    print("Loading test dataset...")
    test_data = load_test_dataset(test_dataset_path)
    
    if not test_data:
        raise ValueError("No test data found")
    
    # Get all unique intents for reference
    all_intents = set()
    for item in test_data:
        all_intents.add(item['act'])
    all_intents = sorted(list(all_intents))
    print(f"Found {len(all_intents)} unique intents: {all_intents}")
    
    # Evaluate
    print("Evaluating model...")
    true_labels = []
    predicted_labels = []
    
    for i, item in enumerate(test_data):
        if i % 10 == 0:
            print(f"Processing sample {i+1}/{len(test_data)}")
        
        dialogue = item['dialogue']
        true_intent = item['act']
        
        # Predict intent
        predicted_intent = predict_intent(model, tokenizer, dialogue)
        
        true_labels.append(true_intent)
        predicted_labels.append(predicted_intent)
        
        # Print some examples
        if i < 5:
            print(f"\nSample {i+1}:")
            print(f"Dialogue: {dialogue[:100]}...")
            print(f"True Intent: {true_intent}")
            print(f"Predicted Intent: {predicted_intent}")
            print(f"Correct: {true_intent == predicted_intent}")
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    # Classification report
    report = classification_report(true_labels, predicted_labels, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=all_intents)
    
    # Per-intent accuracy
    intent_accuracy = {}
    for intent in all_intents:
        intent_indices = [i for i, label in enumerate(true_labels) if label == intent]
        if intent_indices:
            intent_correct = sum(1 for i in intent_indices if predicted_labels[i] == intent)
            intent_accuracy[intent] = intent_correct / len(intent_indices)
    
    results = {
        'overall_accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'intent_accuracy': intent_accuracy,
        'true_labels': true_labels,
        'predicted_labels': predicted_labels,
        'all_intents': all_intents
    }
    
    return results

def print_results(results: Dict):
    """Print evaluation results in a readable format"""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nOverall Accuracy: {results['overall_accuracy']:.4f} ({results['overall_accuracy']*100:.2f}%)")
    
    print(f"\nPer-Intent Accuracy:")
    for intent, acc in results['intent_accuracy'].items():
        print(f"  {intent}: {acc:.4f} ({acc*100:.2f}%)")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(results['true_labels'], results['predicted_labels']))
    
    # Show some examples of mistakes
    print(f"\nSample Mistakes:")
    mistake_count = 0
    for i, (true, pred) in enumerate(zip(results['true_labels'], results['predicted_labels'])):
        if true != pred and mistake_count < 5:
            print(f"  Sample {i+1}: True={true}, Predicted={pred}")
            mistake_count += 1

def main():
    parser = argparse.ArgumentParser(description="Evaluate finetuned Qwen1.5 intent classification model")
    parser.add_argument("--model_path", type=str, default="./qwen_intent_finetuned",
                       help="Path to the finetuned model")
    parser.add_argument("--test_dataset_path", type=str, default="intent_classification_test_dataset",
                       help="Path to the test dataset directory")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json",
                       help="Output file to save results")
    
    args = parser.parse_args()
    
    # Evaluate model
    results = evaluate_model(args.model_path, args.test_dataset_path)
    
    # Print results
    print_results(results)
    
    # Save results
    print(f"\nSaving results to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Evaluation completed!")

if __name__ == "__main__":
    main() 