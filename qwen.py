import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def classify_sentiment(text, model, tokenizer, device="mps"):
    """
    Classify the sentiment of text from SST-2 dataset using Qwen model.
    Returns 0 for negative sentiment, 1 for positive sentiment.
    """
    prompt = f"Classify the sentiment of the following text as positive or negative. Answer with only the number: 1 for positive or 0 for negative.\n\nText: {text}\n\nLabel:"
    enable_thinking = False
    messages = [
        {"role": "user", "content": prompt}
    ]
    formatted_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    
    model_inputs = tokenizer([formatted_input], return_tensors="pt").to(device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=100)
    output_ids = generated_ids[0][len(model_inputs["input_ids"][0]):].tolist()
    output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    print(output)
    # Extract the label from the output
    if "1" in output.split() or "positive" in output.lower():
        return 1
    else:
        return 0  # Default to 0 (negative) if unable to clearly detect positive

def main():
    model_name = "Qwen/Qwen3-0.6B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype='auto',
        device_map="auto"
    )
    
    # Example: Using the model for SST-2 classification
    sst2_examples = [
        # "a masterpiece four years in the making",  # Positive example
        "unfocused and overly complicated",        # Negative example
    ]
    
    for text in sst2_examples:
        label = classify_sentiment(text, model, tokenizer)
        print(f"Text: '{text}'")
        print(f"Sentiment: {label} ({'positive' if label == 1 else 'negative'})")
        print()

if __name__ == "__main__":
    main()
