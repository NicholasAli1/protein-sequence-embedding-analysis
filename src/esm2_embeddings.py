"""
ESM-2 Embedding Generation
Meta's ESM-2 is currently the state-of-the-art protein language model,
outperforming ProtBERT on most downstream tasks.
"""

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import os


class ESM2Embedder:
    """Generate embeddings using Meta's ESM-2 model (Mac-compatible)."""
    
    def __init__(self, model_name='esm2_t33_650M_UR50D', device=None):
        """
        Initialize ESM-2 model.
        
        Args:
            model_name: ESM-2 model variant
                - esm2_t33_650M_UR50D (650M params, 1280-dim, recommended)
                - esm2_t30_150M_UR50D (150M params, 640-dim, faster)
                - esm2_t12_35M_UR50D (35M params, 480-dim, fastest)
                - esm2_t36_3B_UR50D (3B params, 2560-dim, best quality)
            device: Device for computation ('cuda', 'mps', or 'cpu')
                    If None, automatically selects best available device
        """
        # Auto-detect best device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'  # Apple Silicon GPU
            else:
                device = 'cpu'
        
        self.device = device
        self.model_name = model_name
        
        print(f"Loading ESM-2 model ({model_name}) on {device}...")
        if device == 'mps':
            print("🍎 Using Apple Silicon (MPS) acceleration")
        print("📦 Downloading from Meta AI (one-time download)\n")
        
        try:
            import esm
            
            # Load model using direct method (Mac-compatible)
            if model_name == "esm2_t33_650M_UR50D":
                self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            elif model_name == "esm2_t30_150M_UR50D":
                self.model, self.alphabet = esm.pretrained.esm2_t30_150M_UR50D()
            elif model_name == "esm2_t12_35M_UR50D":
                self.model, self.alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            elif model_name == "esm2_t36_3B_UR50D":
                self.model, self.alphabet = esm.pretrained.esm2_t36_3B_UR50D()
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            self.model = self.model.eval().to(device)
            self.batch_converter = self.alphabet.get_batch_converter()
            
            # Print model info
            param_count = sum(p.numel() for p in self.model.parameters()) / 1e6
            print("✅ Model loaded successfully!")
            print(f"   Model: {model_name}")
            print(f"   Parameters: {param_count:.1f}M")
            print(f"   Embedding dim: {self.model.embed_dim}")
            print(f"   Device: {self.device}\n")
            
        except ImportError:
            print("❌ Error: fair-esm not installed")
            print("   Install with: pip install fair-esm")
            raise
    
    def get_embedding(self, sequence, layer_idx=-1):
        """
        Generate ESM-2 embedding for a single sequence.
        
        Args:
            sequence: Amino acid sequence string
            layer_idx: Which layer to extract (-1 for last layer)
            
        Returns:
            numpy array - mean-pooled sequence representation
        """
        # Prepare batch
        data = [("protein", sequence)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        
        # Get representations
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[layer_idx])
            token_representations = results["representations"][layer_idx]
        
        # Mean pool over sequence length (excluding BOS/EOS tokens)
        sequence_repr = token_representations[0, 1:len(sequence)+1].mean(0)
        
        return sequence_repr.cpu().numpy()
    
    def get_per_token_embeddings(self, sequence, layer_idx=-1):
        """
        Get per-residue embeddings for attention visualization.
        
        Args:
            sequence: Amino acid sequence string
            layer_idx: Which layer to extract
            
        Returns:
            numpy array of shape (seq_len, embedding_dim)
        """
        data = [("protein", sequence)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[layer_idx])
            token_representations = results["representations"][layer_idx]
        
        # Return per-token embeddings (excluding BOS/EOS)
        return token_representations[0, 1:len(sequence)+1].cpu().numpy()
    
    def embed_sequences(self, sequences, sequence_ids=None):
        """
        Generate ESM-2 embeddings for multiple sequences.
        
        Args:
            sequences: List of amino acid sequences
            sequence_ids: Optional list of sequence identifiers
            
        Returns:
            Dictionary with embeddings and metadata
        """
        if sequence_ids is None:
            sequence_ids = [f"seq_{i}" for i in range(len(sequences))]
        
        print("Generating ESM-2 embeddings...")
        embeddings = []
        
        for seq in tqdm(sequences, desc="ESM-2"):
            emb = self.get_embedding(seq)
            embeddings.append(emb)
        
        embeddings_array = np.array(embeddings)
        
        embeddings_data = {
            'sequence_ids': sequence_ids,
            'sequences': sequences,
            'esm2': embeddings_array
        }
        
        print(f"\n✅ ESM-2 embeddings shape: {embeddings_array.shape}")
        
        return embeddings_data
    
    def save_embeddings(self, embeddings_data, output_path):
        """Save embeddings to disk."""
        np.savez_compressed(f"{output_path}.npz", **embeddings_data)
        print(f"💾 Embeddings saved to {output_path}.npz")


def main():
    """Generate ESM-2 embeddings for example sequences."""
    
    print("\n" + "="*70)
    print("ESM-2 EMBEDDING GENERATION")
    print("="*70)
    print("\n🧬 Using Meta's ESM-2 (State-of-the-Art Protein Language Model)\n")
    
    # Load sequence data
    print("Loading sequence data...")
    df = pd.read_csv('../example_sequences.csv')
    sequences = df['sequence'].tolist()
    sequence_ids = df['sequence_id'].tolist()
    
    print(f"Loaded {len(sequences)} sequences\n")
    
    # Initialize embedder (auto-detects best device: CUDA, MPS, or CPU)
    embedder = ESM2Embedder(model_name='esm2_t33_650M_UR50D', device=None)
    
    # Generate embeddings
    embeddings_data = embedder.embed_sequences(sequences, sequence_ids)
    
    # Create output directory
    os.makedirs('../data', exist_ok=True)
    
    # Save embeddings
    embedder.save_embeddings(embeddings_data, '../data/embeddings_esm2')
    
    print("\n" + "="*70)
    print("✅ ESM-2 EMBEDDING GENERATION COMPLETE!")
    print("="*70)
    print("\nTo use ESM-2 embeddings in the pipeline:")
    print("  1. Update regression_model.py to load embeddings_esm2.npz")
    print("  2. Set embedding_type='esm2' in the predictor")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
