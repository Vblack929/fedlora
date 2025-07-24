import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig

def load_model(model_path):
    """Load the fine-tuned LoRA model and tokenizer"""
    # Load LoRA config
    config = PeftConfig.from_pretrained(model_path)
    
    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name_or_path,
        num_labels=2,  # Binary classification
        trust_remote_code=True,
        device_map="auto"
    )
    
    # Load LoRA model
    model = PeftModel.from_pretrained(model, model_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer

def predict_sentiment(text, model, tokenizer):
    """Predict sentiment for a given text"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
    
    sentiment = "positive" if predicted_class == 1 else "negative"
    confidence = predictions[0][predicted_class].item()
    
    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "class": predicted_class
    }

def main():
    # Path to the fine-tuned model
    model_path = "./results/qwen-sst2-lora"
    
    # Load model and tokenizer
    model, tokenizer = load_model(model_path)
    
    # Test examples
    test_examples = [
        "This movie was really good.",
        "I hated the ending of that book.",
        "The restaurant was okay, nothing special.",
        "The performance was absolutely brilliant!",
        "What a waste of time and money."
    ]
    
    # Get predictions
    print("Sentiment Analysis Results:")
    print("-" * 50)
    for example in test_examples:
        result = predict_sentiment(example, model, tokenizer)
        print(f"Input: {example}")
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.4f})")
        print("-" * 50)

if __name__ == "__main__":
    main() 