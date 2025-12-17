import discord
from discord.ext import commands
import json
import os
from dotenv import load_dotenv
from datetime import datetime

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix='!', intents=intents)
load_dotenv()
BOT_TOKEN = os.getenv('BOT_TOKEN')

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')
    print(f'Bot is in {len(bot.guilds)} server(s)')

@bot.command(name='scrape')
async def scrape_messages(ctx, user: discord.Member, limit: int = 1000):
    """
    Scrape messages from a specific user
    Usage: !scrape @username [optional_limit]
    """
    await ctx.send(f"Starting to scrape messages from {user.name}... This may take a moment.")
    
    all_messages = []
    
    # Iterate through all text channels in the server
    for channel in ctx.guild.text_channels:
        try:
            # Check if bot has permission to read this channel
            if not channel.permissions_for(ctx.guild.me).read_message_history:
                continue
            
            # Fetch message history
            async for message in channel.history(limit=limit):
                if message.author.id == user.id:
                    # Store relevant message data
                    all_messages.append({
                        'content': message.content,
                        'timestamp': message.created_at.isoformat(),
                        'channel': channel.name,
                        'message_id': message.id
                    })
        
        except discord.Forbidden:
            print(f"No access to channel: {channel.name}")
            continue
    
    # Sort by timestamp
    all_messages.sort(key=lambda x: x['timestamp'])
    
    # Save to file
    filename = f"messages_{user.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_messages, f, indent=2, ensure_ascii=False)
    
    file_size = os.path.getsize(filename) / 1024  # Size in KB
    
    await ctx.send(
        f"✅ Scraping complete!\n"
        f"Found {len(all_messages)} messages from {user.name}\n"
        f"Saved to: {filename}\n"
        f"File size: {file_size:.2f} KB"
    )

@bot.command(name='test')
async def test_scrape(ctx, limit: int = 50):
    """
    Test scrape with a small limit on current channel only
    Usage: !test [optional_limit]
    """
    await ctx.send(f"Testing scrape of last {limit} messages in this channel...")
    
    messages = []
    async for message in ctx.channel.history(limit=limit):
        messages.append({
            'author': message.author.name,
            'content': message.content[:50],  # First 50 chars only for preview
            'timestamp': message.created_at.isoformat()
        })
    
    await ctx.send(f"Found {len(messages)} messages. Check console for preview.")
    print(json.dumps(messages[:5], indent=2))  # Print first 5 to console

# Replace 'YOUR_BOT_TOKEN' with your actual bot token
bot.run()