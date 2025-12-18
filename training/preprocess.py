import json
import re
import os
from pathlib import Path

def clean_message(text):
    """Clean individual message text"""
    # Remove Discord mentions (@user, @role, #channel)
    text = re.sub(r'<@!?\d+>', '', text)
    text = re.sub(r'<@&\d+>', '', text)
    text = re.sub(r'<#\d+>', '', text)
    
    # Remove custom emoji (<:name:id>)
    text = re.sub(r'<a?:\w+:\d+>', '', text)
    
    # Optionally remove URLs (uncomment if you want)
    # text = re.sub(r'http[s]?://\S+', '', text)
    
    # Clean up extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()

def should_include_message(message):
    """Filter out messages we don't want to train on"""
    content = message['content']
    
    # Skip empty messages
    if not content or len(content.strip()) == 0:
        return False
    
    # Skip very short messages (adjust threshold as needed)
    if len(content.strip()) < 3:
        return False
    
    # Skip bot commands
    if content.startswith('!') or content.startswith('/'):
        return False
    
    # Skip messages that are only links
    if content.startswith('http') and len(content.split()) == 1:
        return False
    
    # Skip messages that are only emoji or mentions
    cleaned = clean_message(content)
    if len(cleaned) < 3:
        return False
    
    return True

def preprocess_json_to_text(json_path, output_path, format_style='simple'):
    """
    Convert JSON messages to training text
    
    Args:
        json_path: Path to scraped JSON file
        output_path: Path for output text file
        format_style: 'simple' or 'chat' format
    """
    # Load messages
    with open(json_path, 'r', encoding='utf-8') as f:
        messages = json.load(f)
    
    print(f"Loaded {len(messages)} messages")
    
    # Filter and clean messages
    cleaned_messages = []
    for msg in messages:
        if should_include_message(msg):
            cleaned_text = clean_message(msg['content'])
            if cleaned_text:
                cleaned_messages.append(cleaned_text)
    
    print(f"After filtering: {len(cleaned_messages)} messages")
    
    # Format for training
    if format_style == 'simple':
        # Simple format: one message per line with separator
        output_text = '<|endoftext|>'.join(cleaned_messages)
    
    elif format_style == 'separated':
        # Messages separated by double newline
        output_text = '\n\n'.join(cleaned_messages)
    
    else:
        raise ValueError(f"Unknown format_style: {format_style}")
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output_text)
    
    file_size = os.path.getsize(output_path) / 1024
    print(f"Saved {len(cleaned_messages)} messages to {output_path}")
    print(f"Output file size: {file_size:.2f} KB")
    
    # Print some stats
    total_chars = sum(len(msg) for msg in cleaned_messages)
    avg_length = total_chars / len(cleaned_messages) if cleaned_messages else 0
    print(f"Average message length: {avg_length:.1f} characters")
    
    return cleaned_messages

def preprocess_all_json_files(data_dir='data', output_dir='data/processed'):
    """Process all JSON files in data directory"""
    Path(output_dir).mkdir(exist_ok=True)
    
    json_files = list(Path(data_dir).glob('messages_*.json'))
    
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return
    
    print(f"Found {len(json_files)} JSON file(s)")
    
    for json_file in json_files:
        print(f"\n Processing {json_file.name}...")
        
        # Create output filename
        output_name = json_file.stem + '_cleaned.txt'
        output_path = Path(output_dir) / output_name
        
        preprocess_json_to_text(json_file, output_path, format_style='simple')

if __name__ == '__main__':
    # Process all JSON files in data/ directory
    preprocess_all_json_files()
    
    # Or process a specific file:
    # preprocess_json_to_text('data/messages_friend_20241217.json', 
    #                         'data/processed/training_data.txt')