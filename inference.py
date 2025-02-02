import torch
from mingpt.bpe import BPETokenizer
from custom_gpt import CustomGPT

def load_trained_model(model_path='gpt_model.pt', device='cuda'):
    # Model configuration - must match training configuration EXACTLY
    model_config = CustomGPT.get_default_config()
    model_config.model_type = 'gpt-mini'  # This has 6 layers and n_embd=192
    model_config.block_size = 128
    model_config.vocab_size = 50257
    model_config.num_harmonics = 8  # Match training value
    
    # Dropout settings (matching training)
    model_config.embd_pdrop = 0.0
    model_config.resid_pdrop = 0.0
    model_config.attn_pdrop = 0.0
    
    # Initialize model
    model = CustomGPT(model_config)
    
    # Load the saved state with weights_only=True
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
    
    model = model.to(device)
    model.eval()
    return model

def generate_text(model, prompt, max_tokens=50, temperature=0.7, top_k=40):
    """
    Generate text from a prompt.
    """
    tokenizer = BPETokenizer()
    device = next(model.parameters()).device
    
    # Tokenize the prompt and add batch dimension if needed
    try:
        tokens = tokenizer(prompt)
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        input_ids = tokens.to(device)
        print(f"Input tensor shape after tokenization: {input_ids.shape}")
    except Exception as e:
        print(f"Error during tokenization: {str(e)}")
        raise
    
    # Generate
    try:
        with torch.no_grad():
            output_sequence = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_k=top_k
            )
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        raise
    
    # Decode the generated sequence
    try:
        # Ensure we're working with a tensor before converting to list
        if torch.is_tensor(output_sequence):
            # Get first sequence from batch and convert to tensor
            sequence_tensor = output_sequence[0].clone().detach()
            # Convert to CPU if needed
            if sequence_tensor.is_cuda:
                sequence_tensor = sequence_tensor.cpu()
            generated_text = tokenizer.decode(sequence_tensor)
        else:
            raise ValueError("Expected output_sequence to be a tensor")
        return generated_text
    except Exception as e:
        print(f"Error during decoding: {str(e)}")
        raise

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        model = load_trained_model('gpt_model.pt', device)
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        exit(1)
    
    test_prompts = [
        "The history of the Roman Empire began when"
    ]
    
    print("\nGenerating text samples...")
    print("-" * 50)
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        try:
            generated = generate_text(
                model,
                prompt,
                max_tokens=50,
                temperature=0.6,
                top_k=20
            )
            print(f"Generated: {generated}")
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            import traceback
            print(traceback.format_exc())
        print("-" * 50)
