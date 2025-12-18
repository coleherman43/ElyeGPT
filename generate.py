import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

def generate_text(
    model_path='models/models/elye_gpt',
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
    
    # Convert to absolute path to avoid huggingface repo ID confusion
    if not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)
    
    print(f"Loading model from {model_path}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)
    
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
    results = []
    for i, output in enumerate(outputs):
        full_text = tokenizer.decode(output, skip_special_tokens=True)
        
        # Extract only the newly generated text (remove the prompt)
        if prompt:
            # Remove the prompt from the beginning
            prompt_clean = prompt.replace('<|endoftext|>', '')
            if full_text.startswith(prompt_clean):
                generated_text = full_text[len(prompt_clean):].strip()
            else:
                generated_text = full_text
        else:
            generated_text = full_text
        
        if num_return_sequences > 1:
            print(f"\n--- Response #{i+1} ---")
        print(generated_text)
        print()
        results.append(generated_text)
    
    return results

if __name__ == '__main__':
    import sys
    
    # Check if a prompt was provided as command line argument
    if len(sys.argv) > 1:
        user_input = ' '.join(sys.argv[1:])
    else:
        user_input = input("Enter a prompt (or press Enter to start fresh): ")
    
    # Format the prompt for conversational response
    if user_input:
        # Option 1: Few-shot - add context examples before the user's message
        # This primes the model to respond in your friend's style
        context = """I will stop being pessimistic about my academic and career future 🙏<|endoftext|>I will not relate to Xiu Xiu lyrics!<|endoftext|>I do not suffer from clinical depression ✨<|endoftext|>"""
        prompt = context + user_input + "<|endoftext|>"
    else:
        # Generate from scratch
        prompt = ""
    
    # Generate text
    generate_text(
        prompt=prompt,
        max_length=50,
        temperature=0.9,  # Higher temperature for more personality
        num_return_sequences=3  # Generate 3 different variations
    )
