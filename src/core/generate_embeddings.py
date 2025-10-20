"""
Generate protein sequence embeddings using UniRep and ProtBERT models.
This module extracts latent representations for antibody developability analysis.
"""

import numpy as np
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import joblib
import os

# Try to import UniRep, but make it optional
try:
    from jax_unirep import get_reps
    UNIREP_AVAILABLE = True
except ImportError:
    UNIREP_AVAILABLE = False
    print("⚠️  Warning: jax-unirep not available. UniRep embeddings will be skipped.")
    print("   ProtBERT embeddings will still work! ProtBERT is state-of-the-art anyway.")
    get_reps = None


class SequenceEmbedder:
    """Generate embeddings from protein sequences using multiple models."""
    
    def __init__(self, use_unirep=True, use_protbert=True, device='cpu'):
        """
        Initialize embedding models.
        
        Args:
            use_unirep: Use UniRep model for embeddings
            use_protbert: Use ProtBERT model for embeddings
            device: Device for PyTorch models ('cpu' or 'cuda')
        """
        self.use_unirep = use_unirep and UNIREP_AVAILABLE
        self.use_protbert = use_protbert
        self.device = device
        
        if use_unirep and not UNIREP_AVAILABLE:
            print("⚠️  UniRep requested but not available, skipping...")
        
        # Initialize ProtBERT if requested
        if self.use_protbert:
            print("Loading ProtBERT model...")
            self.protbert_tokenizer = BertTokenizer.from_pretrained(
                "Rostlab/prot_bert", do_lower_case=False
            )
            self.protbert_model = BertModel.from_pretrained("Rostlab/prot_bert")
            self.protbert_model.to(self.device)
            self.protbert_model.eval()
            print("ProtBERT loaded successfully")
    
    def get_unirep_embedding(self, sequence):
        """
        Generate UniRep embedding for a single sequence.
        
        Args:
            sequence: Amino acid sequence string
            
        Returns:
            numpy array of shape (1900,) - UniRep embedding
        """
        # Get hidden state average representation (1900-dimensional)
        h_avg, h_final, c_final = get_reps(sequence)
        return h_avg
    
    def get_protbert_embedding(self, sequence):
        """
        Generate ProtBERT embedding for a single sequence.
        
        Args:
            sequence: Amino acid sequence string
            
        Returns:
            numpy array - ProtBERT embedding (1024-dimensional from pooler output)
        """
        # Add spaces between amino acids as required by ProtBERT
        spaced_sequence = ' '.join(list(sequence))
        
        # Tokenize and encode
        encoded = self.protbert_tokenizer(
            spaced_sequence,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Get embeddings
        with torch.no_grad():
            output = self.protbert_model(**encoded)
            # Use pooler output (CLS token representation)
            embedding = output.pooler_output.cpu().numpy()[0]
        
        return embedding
    
    def embed_sequences(self, sequences, sequence_ids=None):
        """
        Generate embeddings for multiple sequences.
        
        Args:
            sequences: List of amino acid sequences
            sequence_ids: Optional list of sequence identifiers
            
        Returns:
            Dictionary with embeddings and metadata
        """
        if sequence_ids is None:
            sequence_ids = [f"seq_{i}" for i in range(len(sequences))]
        
        embeddings_data = {
            'sequence_ids': sequence_ids,
            'sequences': sequences
        }
        
        # Generate UniRep embeddings
        if self.use_unirep:
            print("Generating UniRep embeddings...")
            unirep_embeddings = []
            for seq in tqdm(sequences, desc="UniRep"):
                emb = self.get_unirep_embedding(seq)
                unirep_embeddings.append(emb)
            embeddings_data['unirep'] = np.array(unirep_embeddings)
            print(f"UniRep embeddings shape: {embeddings_data['unirep'].shape}")
        
        # Generate ProtBERT embeddings
        if self.use_protbert:
            print("Generating ProtBERT embeddings...")
            protbert_embeddings = []
            for seq in tqdm(sequences, desc="ProtBERT"):
                emb = self.get_protbert_embedding(seq)
                protbert_embeddings.append(emb)
            embeddings_data['protbert'] = np.array(protbert_embeddings)
            print(f"ProtBERT embeddings shape: {embeddings_data['protbert'].shape}")
        
        return embeddings_data
    
    def save_embeddings(self, embeddings_data, output_path):
        """
        Save embeddings to disk.
        
        Args:
            embeddings_data: Dictionary containing embeddings
            output_path: Path to save the embeddings (without extension)
        """
        # Save as compressed numpy archive
        np.savez_compressed(
            f"{output_path}.npz",
            **embeddings_data
        )
        print(f"Embeddings saved to {output_path}.npz")
    
    @staticmethod
    def load_embeddings(input_path):
        """
        Load embeddings from disk.
        
        Args:
            input_path: Path to the embeddings file
            
        Returns:
            Dictionary containing embeddings
        """
        data = np.load(input_path, allow_pickle=True)
        embeddings_data = {key: data[key] for key in data.files}
        return embeddings_data


def main():
    """Main function to generate embeddings from example sequences."""
    
    # Load sequence data
    print("Loading sequence data...")
    # Get absolute path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '..', 'datasets', 'example_sequences.csv')
    df = pd.read_csv(data_path)
    sequences = df['sequence'].tolist()
    sequence_ids = df['sequence_id'].tolist()
    
    print(f"Loaded {len(sequences)} sequences")
    
    # Check for GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize embedder
    embedder = SequenceEmbedder(
        use_unirep=True,
        use_protbert=True,
        device=device
    )
    
    # Generate embeddings
    embeddings_data = embedder.embed_sequences(sequences, sequence_ids)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(script_dir, '..', '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save embeddings
    output_path = os.path.join(output_dir, 'embeddings')
    embedder.save_embeddings(embeddings_data, output_path)
    
    print("\n=== Embedding Generation Complete ===")
    print(f"Total sequences processed: {len(sequences)}")
    if 'unirep' in embeddings_data:
        print(f"UniRep embedding dimensions: {embeddings_data['unirep'].shape[1]}")
    if 'protbert' in embeddings_data:
        print(f"ProtBERT embedding dimensions: {embeddings_data['protbert'].shape[1]}")


if __name__ == "__main__":
    main()
