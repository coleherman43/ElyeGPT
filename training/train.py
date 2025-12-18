import torch
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    DataCollatorForLanguageModeling,
    Trainer, 
    TrainingArguments
)
from datasets import load_dataset
from pathlib import Path
import os

def train_model(
    train_file_path,
    output_dir='models/elye_gpt',
    model_name='distilgpt2',  # 'distilgpt2' for smaller/faster
    block_size=128,
    num_train_epochs=50,  # Very aggressive for small datasets
    per_device_train_batch_size=1,  # Smaller batches = more frequent updates
    save_steps=50,
    learning_rate=2e-4  # Higher learning rate for faster adaptation
):
    """
    Fine-tune GPT-2 on friend's messages
    
    Args:
        train_file_path: Path to cleaned text file
        output_dir: Where to save the trained model
        model_name: Base model to fine-tune ('gpt2', 'distilgpt2', etc.)
        block_size: Max sequence length for training
        num_train_epochs: Number of training passes
        per_device_train_batch_size: Batch size (lower if you run out of memory)
        save_steps: Save checkpoint every N steps
        learning_rate: Learning rate for training
    """
    
    print(f"Starting training with {model_name}")
    print(f"Training file: {train_file_path}")
    
    # Check if file exists
    if not os.path.exists(train_file_path):
        raise FileNotFoundError(f"Training file not found: {train_file_path}")
    
    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # GPT-2 doesn't have a pad token by default, so we'll use eos_token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print("Loading and tokenizing dataset...")
    dataset = load_dataset('text', data_files={'train': train_file_path})
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=block_size)
    
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=['text']
    )
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal language modeling, not masked
    )
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check for GPU availability
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = "cpu"
        print("Using CPU")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        save_steps=save_steps,
        save_total_limit=2,  # Only keep last 2 checkpoints
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        warmup_steps=100,
        prediction_loss_only=True,
        use_mps_device=(device == "mps"),  # Enable Apple Silicon GPU if available
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset['train'],
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Training complete!")
    print(f"Model saved to: {output_dir}")
    print(f"\nTo use the model:")
    print(f"  from transformers import GPT2LMHeadModel, GPT2Tokenizer")
    print(f"  model = GPT2LMHeadModel.from_pretrained('{output_dir}')")
    print(f"  tokenizer = GPT2Tokenizer.from_pretrained('{output_dir}')")
    
    return model, tokenizer

if __name__ == '__main__':
    # Example usage - adjust the path to your processed text file
    import sys
    
    if len(sys.argv) > 1:
        train_file = sys.argv[1]
    else:
        # Default to first processed file found
        processed_dir = Path('data/processed')
        text_files = list(processed_dir.glob('*_cleaned.txt'))
        
        if not text_files:
            print("Error: No processed text files found in data/processed/")
            print("Run preprocess.py first to generate training data")
            sys.exit(1)
        
        train_file = str(text_files[0])
        print(f"Using training file: {train_file}")
    
    # Train the model with aggressive settings for small datasets
    train_model(
        train_file_path=train_file,
        output_dir='models/elye_gpt',
        model_name='distilgpt2',
        block_size=128,
        num_train_epochs=15,  # More epochs for small data
        per_device_train_batch_size=4,
        save_steps=50,  # Save more frequently
        learning_rate=1e-4  # Higher learning rate for faster adaptation
    )