#!/usr/bin/env python3
"""
Inference script for the finetuned Qwen1.5 intent classification model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

def load_model(model_path: str):
    """Load the finetuned model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype="auto"
    )
    return model, tokenizer

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
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
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

def main():
    parser = argparse.ArgumentParser(description="Inference with finetuned Qwen1.5 intent classification model")
    parser.add_argument("--model_path", type=str, default="./qwen_intent_finetuned",
                       help="Path to the finetuned model")
    parser.add_argument("--dialogue", type=str, 
                       default="<USER> I need a hotel room in New York.<SYSTEM> What's your preferred star rating?<USER> 4 stars please.<SYSTEM> I found the Marriott Marquis Times Square, a 4-star hotel.",
                       help="Dialogue to classify")
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model(args.model_path)
    
    # Predict intent
    print(f"Dialogue: {args.dialogue}")
    print("Predicting intent...")
    
    intent = predict_intent(model, tokenizer, args.dialogue)
    print(f"Predicted Intent: {intent}")
    
    # Interactive mode
    print("\n" + "="*50)
    print("Interactive mode (type 'quit' to exit):")
    print("="*50)
    
    while True:
        try:
            user_dialogue = input("\nEnter dialogue: ")
            if user_dialogue.lower() == 'quit':
                break
            
            if user_dialogue.strip():
                intent = predict_intent(model, tokenizer, user_dialogue)
                print(f"Predicted Intent: {intent}")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main() 