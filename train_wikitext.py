# train_wikitext.py

import math
import torch
from datasets import load_dataset
from mingpt.trainer import Trainer
from mingpt.bpe import BPETokenizer
from torch.utils.data import Dataset
from custom_gpt import CustomGPT

class WikiTextDataset(Dataset):
    def __init__(self, block_size=1024):
        # Load dataset
        print("Loading WikiText-2 dataset...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        
        # Initialize tokenizer
        self.tokenizer = BPETokenizer()
        self.block_size = block_size
        
        # Concatenate all texts
        text = "\n".join(dataset["train"]["text"])
        
        print("Tokenizing dataset...")
        # Tokenize the entire text
        data = self.tokenizer(text)[0]  # Remove batch dimension
        
        # Create sequences
        print("Creating training sequences...")
        self.examples = []
        for i in range(0, len(data) - block_size):
            self.examples.append(data[i:i + block_size + 1])
        print(f"Created {len(self.examples)} training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        x = self.examples[i][:-1]
        y = self.examples[i][1:]
        return x, y

def train():
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB")
    
    # Model configuration
    model_config = CustomGPT.get_default_config()
    model_config.model_type = 'gpt2-medium'
    model_config.block_size = 128
    model_config.vocab_size = 50257
    model_config.num_harmonics = 8
    
    # Dropout settings
    model_config.embd_pdrop = 0.0
    model_config.resid_pdrop = 0.0
    model_config.attn_pdrop = 0.0
    
    print("Initializing dataset...")
    train_dataset = WikiTextDataset(block_size=model_config.block_size)
    
    print("Initializing model...")
    model = CustomGPT(model_config)
    model = model.to(device)  # Move model to GPU
    
    # Initialize weights
    def init_weights(module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
    model.apply(init_weights)
    
    # Training configuration
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 6e-4
    train_config.max_iters = 10000
    train_config.batch_size = 64
    train_config.num_workers = 2  # Increased for GPU training
    train_config.grad_norm_clip = 0.5
    train_config.device = device  # Set device in config
    
    # Learning rate warmup
    warmup_iters = 100
    
    def get_lr(iter_num):
        if iter_num < warmup_iters:
            return train_config.learning_rate * (iter_num / warmup_iters)
        decay_ratio = (iter_num - warmup_iters) / (train_config.max_iters - warmup_iters)
        return train_config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    
    # Add training callbacks for monitoring
    def batch_end_callback(trainer):
        # Update learning rate
        if hasattr(trainer, 'optimizer'):
            lr = get_lr(trainer.iter_num)
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = lr
        
        if trainer.iter_num % 100 == 0:
            curr_lr = trainer.optimizer.param_groups[0]['lr']
            print(f"iter {trainer.iter_num}: loss {trainer.loss.item():.4f}, lr {curr_lr:.2e}")
            if device.type == "cuda":
                print(f"GPU Memory: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB")
    
    print("Starting training...")
    trainer = Trainer(train_config, model, train_dataset)
    trainer.set_callback('on_batch_end', batch_end_callback)
    
    trainer.run()
    
    print("Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'loss': trainer.loss,
        'iter_num': trainer.iter_num,
    }, 'gpt_model.pt')
    print("Training complete!")

if __name__ == '__main__':
    train()
