import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(
    model_path='models/friend_bot/models/friend_bot',
    prompt='',
    max_length=100,
    temperature=0.8,
    top_p=0.9,
    num_return_sequences=1
):
    """
    Generate text using the trained model
    
    Args:
        model_path: Path to the trained model
        prompt: Starting text (leave empty to start fresh)
        max_length: Maximum length of generated text
        temperature: Higher = more random, lower = more focused (0.7-1.0 recommended)
        top_p: Nucleus sampling parameter (0.9 recommended)
        num_return_sequences: Number of different outputs to generate
    """
    
    print(f"Loading model from {model_path}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"Using device: {device}")
    print(f"\nPrompt: '{prompt}'")
    print("=" * 50)
    
    # Encode the prompt
    if prompt:
        encoded = tokenizer(prompt, return_tensors='pt')
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
    else:
        # Start with just the beginning of sequence token
        input_ids = torch.tensor([[tokenizer.bos_token_id]]).to(device)
        attention_mask = torch.ones_like(input_ids).to(device)
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2,  # Avoid repeating phrases
        )
    
    # Decode and print results
    for i, output in enumerate(outputs):
        text = tokenizer.decode(output, skip_special_tokens=True)
        if num_return_sequences > 1:
            print(f"\n--- Generated Text #{i+1} ---")
        print(text)
        print()
    
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

if __name__ == '__main__':
    import sys
    
    # Check if a prompt was provided as command line argument
    if len(sys.argv) > 1:
        prompt = ' '.join(sys.argv[1:])
    else:
        prompt = input("Enter a prompt (or press Enter to start fresh): ")
    
    # Generate text
    generate_text(
        prompt=prompt,
        max_length=50,
        temperature=0.8,
        num_return_sequences=3  # Generate 3 different variations
    )
