"""
Attention Visualization for Protein Language Models
Visualize which parts of the sequence the model attends to.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import BertModel, BertTokenizer
import plotly.graph_objects as go
import os


class AttentionVisualizer:
    """Visualize attention patterns in protein language models."""
    
    def __init__(self, model_name='Rostlab/prot_bert', device='cpu'):
        """
        Initialize attention visualizer.
        
        Args:
            model_name: HuggingFace model name
            device: Device for computation
        """
        self.device = device
        
        print(f"Loading {model_name} for attention visualization...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = BertModel.from_pretrained(model_name, output_attentions=True)
        self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded\n")
    
    def get_attention_weights(self, sequence):
        """
        Extract attention weights from all layers.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            attention_weights: List of attention matrices for each layer
            tokens: List of tokens
        """
        # Add spaces between amino acids
        spaced_sequence = ' '.join(list(sequence))
        
        # Tokenize
        encoded = self.tokenizer(
            spaced_sequence,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        
        tokens = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Get attention weights
        with torch.no_grad():
            outputs = self.model(**encoded)
            attentions = outputs.attentions  # Tuple of (batch, num_heads, seq_len, seq_len)
        
        # Convert to numpy and average over heads
        attention_weights = []
        for layer_attention in attentions:
            # Average over attention heads
            avg_attention = layer_attention[0].mean(dim=0).cpu().numpy()
            attention_weights.append(avg_attention)
        
        return attention_weights, tokens
    
    def plot_attention_heatmap(self, sequence, layer_idx=-1, output_dir='plots/attention'):
        """
        Plot attention heatmap for a specific layer.
        
        Args:
            sequence: Amino acid sequence
            layer_idx: Which layer to visualize (-1 for last layer)
            output_dir: Directory to save plot
        """
        os.makedirs(output_dir, exist_ok=True)
        
        attention_weights, tokens = self.get_attention_weights(sequence)
        attention_matrix = attention_weights[layer_idx]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Plot heatmap
        sns.heatmap(
            attention_matrix,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='viridis',
            cbar_kws={'label': 'Attention Weight'},
            ax=ax
        )
        
        ax.set_title(f'Attention Heatmap - Layer {layer_idx}\nSequence: {sequence[:50]}...')
        ax.set_xlabel('Key Tokens')
        ax.set_ylabel('Query Tokens')
        
        plt.tight_layout()
        
        filename = f'attention_layer{layer_idx}_heatmap.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {filename}")
    
    def plot_attention_head_view(self, sequence, layer_idx=-1, output_dir='plots/attention'):
        """
        Visualize attention for all heads in a layer.
        
        Args:
            sequence: Amino acid sequence
            layer_idx: Which layer to visualize
            output_dir: Directory to save plot
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get per-head attention
        spaced_sequence = ' '.join(list(sequence))
        encoded = self.tokenizer(spaced_sequence, return_tensors='pt', 
                                truncation=True, max_length=512)
        tokens = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = self.model(**encoded)
            layer_attention = outputs.attentions[layer_idx][0].cpu().numpy()  # (num_heads, seq_len, seq_len)
        
        num_heads = layer_attention.shape[0]
        
        # Plot all heads
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        for head_idx in range(min(num_heads, 12)):
            sns.heatmap(
                layer_attention[head_idx],
                cmap='viridis',
                cbar=True,
                ax=axes[head_idx],
                xticklabels=False,
                yticklabels=False
            )
            axes[head_idx].set_title(f'Head {head_idx + 1}')
        
        # Hide unused subplots
        for idx in range(num_heads, 12):
            axes[idx].axis('off')
        
        plt.suptitle(f'Attention Heads - Layer {layer_idx}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        filename = f'attention_layer{layer_idx}_heads.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {filename}")
    
    def plot_attention_rollout(self, sequence, output_dir='plots/attention'):
        """
        Compute and visualize attention rollout across all layers.
        Attention rollout shows cumulative attention flow.
        
        Args:
            sequence: Amino acid sequence
            output_dir: Directory to save plot
        """
        os.makedirs(output_dir, exist_ok=True)
        
        attention_weights, tokens = self.get_attention_weights(sequence)
        
        # Compute attention rollout
        rollout = np.eye(attention_weights[0].shape[0])
        
        for layer_attention in attention_weights:
            # Add residual connection
            layer_attention = layer_attention + np.eye(layer_attention.shape[0])
            # Normalize
            layer_attention = layer_attention / layer_attention.sum(axis=-1, keepdims=True)
            # Multiply
            rollout = np.matmul(rollout, layer_attention)
        
        # Focus on CLS token attention
        cls_attention = rollout[0, :]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        # Plot 1: Attention heatmap
        sns.heatmap(
            rollout,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='viridis',
            cbar_kws={'label': 'Attention Flow'},
            ax=ax1
        )
        ax1.set_title('Attention Rollout - Full Matrix')
        ax1.set_xlabel('Target Tokens')
        ax1.set_ylabel('Source Tokens')
        
        # Plot 2: CLS token attention to each position
        positions = list(range(len(tokens)))
        ax2.bar(positions, cls_attention, alpha=0.7, color='steelblue')
        ax2.set_xticks(positions)
        ax2.set_xticklabels(tokens, rotation=90)
        ax2.set_xlabel('Sequence Position')
        ax2.set_ylabel('Attention Weight from CLS')
        ax2.set_title('CLS Token Attention Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = 'attention_rollout.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {filename}")
        
        return rollout, tokens
    
    def create_interactive_attention(self, sequence, layer_idx=-1, output_dir='plots/attention'):
        """
        Create interactive Plotly attention visualization.
        
        Args:
            sequence: Amino acid sequence
            layer_idx: Which layer to visualize
            output_dir: Directory to save plot
        """
        os.makedirs(output_dir, exist_ok=True)
        
        attention_weights, tokens = self.get_attention_weights(sequence)
        attention_matrix = attention_weights[layer_idx]
        
        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=attention_matrix,
            x=tokens,
            y=tokens,
            colorscale='Viridis',
            hovertemplate='Query: %{y}<br>Key: %{x}<br>Attention: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Interactive Attention Heatmap - Layer {layer_idx}',
            xaxis_title='Key Tokens',
            yaxis_title='Query Tokens',
            width=900,
            height=800,
            font=dict(size=10)
        )
        
        filename = f'attention_layer{layer_idx}_interactive.html'
        fig.write_html(os.path.join(output_dir, filename))
        
        print(f"✅ Saved: {filename}")


def analyze_sequence_attention(sequence, sequence_id='protein', output_dir='plots/attention'):
    """
    Complete attention analysis for a protein sequence.
    
    Args:
        sequence: Amino acid sequence
        sequence_id: Identifier for the sequence
        output_dir: Directory to save plots
    """
    print("\n" + "="*70)
    print(f"ATTENTION VISUALIZATION ANALYSIS")
    print("="*70)
    print(f"\nAnalyzing sequence: {sequence_id}")
    print(f"Length: {len(sequence)} amino acids\n")
    
    # Initialize visualizer
    visualizer = AttentionVisualizer()
    
    # Generate all visualizations
    print("Generating visualizations...")
    visualizer.plot_attention_heatmap(sequence, layer_idx=-1, output_dir=output_dir)
    visualizer.plot_attention_head_view(sequence, layer_idx=-1, output_dir=output_dir)
    visualizer.plot_attention_rollout(sequence, output_dir=output_dir)
    visualizer.create_interactive_attention(sequence, layer_idx=-1, output_dir=output_dir)
    
    print("\n" + "="*70)
    print("✅ ATTENTION ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nPlots saved to: {output_dir}/")
    print("\nInterpretation:")
    print("  • Bright regions = high attention")
    print("  • Attention rollout = cumulative attention flow")
    print("  • CLS token attention = sequence-level importance")
    print("  • Different heads capture different patterns")
    print("="*70 + "\n")


def main():
    """Main function to visualize attention for example sequences."""
    
    import pandas as pd
    
    # Load sequences
    df = pd.read_csv('../../datasets/example_sequences.csv')
    
    # Analyze first few sequences
    for idx in range(min(3, len(df))):
        sequence = df.iloc[idx]['sequence']
        sequence_id = df.iloc[idx]['sequence_id']
        
        analyze_sequence_attention(
            sequence,
            sequence_id,
            output_dir=f'../../plots/attention/{sequence_id}'
        )


if __name__ == "__main__":
    main()
