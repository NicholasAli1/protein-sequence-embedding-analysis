"""
Paired Heavy/Light Chain Analysis for Antibodies
Analyze antibody developability considering both chains together.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import os


class PairedChainAnalyzer:
    """Analyze paired antibody heavy and light chains."""
    
    def __init__(self, embedding_model='Rostlab/prot_bert', device='cpu'):
        """
        Initialize paired chain analyzer.
        
        Args:
            embedding_model: Model for generating embeddings
            device: Device for computation
        """
        self.device = device
        
        print("Loading embedding model for paired chains...")
        self.tokenizer = BertTokenizer.from_pretrained(embedding_model, do_lower_case=False)
        self.model = BertModel.from_pretrained(embedding_model)
        self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded\n")
    
    def get_chain_embedding(self, sequence):
        """Generate embedding for a single chain."""
        spaced_sequence = ' '.join(list(sequence))
        encoded = self.tokenizer(
            spaced_sequence,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        with torch.no_grad():
            output = self.model(**encoded)
            embedding = output.pooler_output.cpu().numpy()[0]
        
        return embedding
    
    def get_paired_embedding(self, heavy_chain, light_chain, concatenation='concat'):
        """
        Generate combined embedding for paired chains.
        
        Args:
            heavy_chain: Heavy chain sequence
            light_chain: Light chain sequence
            concatenation: How to combine ('concat', 'mean', 'attention')
            
        Returns:
            Combined embedding
        """
        heavy_emb = self.get_chain_embedding(heavy_chain)
        light_emb = self.get_chain_embedding(light_chain)
        
        if concatenation == 'concat':
            # Simple concatenation
            paired_emb = np.concatenate([heavy_emb, light_emb])
        elif concatenation == 'mean':
            # Average of both chains
            paired_emb = (heavy_emb + light_emb) / 2
        elif concatenation == 'weighted':
            # Weighted average (heavy chain typically more important)
            paired_emb = 0.6 * heavy_emb + 0.4 * light_emb
        elif concatenation == 'interaction':
            # Include interaction term
            interaction = heavy_emb * light_emb
            paired_emb = np.concatenate([heavy_emb, light_emb, interaction])
        else:
            raise ValueError(f"Unknown concatenation method: {concatenation}")
        
        return paired_emb
    
    def analyze_chain_contributions(self, heavy_chain, light_chain, 
                                    metric_predictor=None, metric_scaler=None):
        """
        Analyze individual chain contributions to developability.
        
        Args:
            heavy_chain: Heavy chain sequence
            light_chain: Light chain sequence
            metric_predictor: Trained model for prediction
            metric_scaler: Fitted scaler
            
        Returns:
            Dictionary of contribution scores
        """
        # Get individual embeddings
        heavy_emb = self.get_chain_embedding(heavy_chain)
        light_emb = self.get_chain_embedding(light_chain)
        
        # Get paired embedding
        paired_emb = self.get_paired_embedding(heavy_chain, light_chain, 'concat')
        
        if metric_predictor and metric_scaler:
            # Predict with each configuration
            heavy_only = np.concatenate([heavy_emb, np.zeros_like(light_emb)])
            light_only = np.concatenate([np.zeros_like(heavy_emb), light_emb])
            
            predictions = {
                'paired': metric_predictor.predict(metric_scaler.transform([paired_emb]))[0],
                'heavy_only': metric_predictor.predict(metric_scaler.transform([heavy_only]))[0],
                'light_only': metric_predictor.predict(metric_scaler.transform([light_only]))[0]
            }
            
            # Compute contributions
            contributions = {
                'heavy_contribution': predictions['heavy_only'] - predictions['light_only'],
                'light_contribution': predictions['light_only'] - predictions['heavy_only'],
                'synergy': predictions['paired'] - (predictions['heavy_only'] + predictions['light_only'])
            }
        else:
            # Just return similarity metrics
            contributions = {
                'heavy_light_similarity': np.dot(heavy_emb, light_emb) / (
                    np.linalg.norm(heavy_emb) * np.linalg.norm(light_emb)
                ),
                'heavy_norm': np.linalg.norm(heavy_emb),
                'light_norm': np.linalg.norm(light_emb)
            }
        
        return contributions
    
    def visualize_chain_interaction(self, heavy_chain, light_chain, 
                                    output_dir='plots/paired_chains'):
        """
        Visualize heavy/light chain interaction patterns.
        
        Args:
            heavy_chain: Heavy chain sequence
            light_chain: Light chain sequence
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get embeddings
        heavy_emb = self.get_chain_embedding(heavy_chain)
        light_emb = self.get_chain_embedding(light_chain)
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 10))
        
        # Plot 1: Embedding comparison
        ax1 = plt.subplot(2, 2, 1)
        positions = np.arange(min(len(heavy_emb), 100))  # Show first 100 dims
        ax1.plot(positions, heavy_emb[:100], label='Heavy Chain', alpha=0.7, linewidth=2)
        ax1.plot(positions, light_emb[:100], label='Light Chain', alpha=0.7, linewidth=2)
        ax1.set_xlabel('Embedding Dimension')
        ax1.set_ylabel('Value')
        ax1.set_title('Embedding Comparison (First 100 dims)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Correlation heatmap
        ax2 = plt.subplot(2, 2, 2)
        # Reshape for correlation
        corr_matrix = np.corrcoef(heavy_emb[:100], light_emb[:100])
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   xticklabels=['Heavy', 'Light'], yticklabels=['Heavy', 'Light'], ax=ax2)
        ax2.set_title('Chain Embedding Correlation')
        
        # Plot 3: Element-wise product (interaction)
        ax3 = plt.subplot(2, 2, 3)
        interaction = heavy_emb[:100] * light_emb[:100]
        ax3.bar(positions, interaction, alpha=0.7, color='purple')
        ax3.set_xlabel('Embedding Dimension')
        ax3.set_ylabel('Interaction Strength')
        ax3.set_title('Heavy-Light Interaction Pattern')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Sequence length comparison
        ax4 = plt.subplot(2, 2, 4)
        chains = ['Heavy', 'Light']
        lengths = [len(heavy_chain), len(light_chain)]
        colors_bar = ['steelblue', 'coral']
        ax4.bar(chains, lengths, color=colors_bar, alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Sequence Length')
        ax4.set_title('Chain Length Comparison')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add text annotations
        for i, (chain, length) in enumerate(zip(chains, lengths)):
            ax4.text(i, length + 5, str(length), ha='center', fontweight='bold')
        
        plt.suptitle('Paired Chain Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        filename = 'chain_interaction.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {filename}")
    
    def predict_pairing_compatibility(self, heavy_chains, light_chains):
        """
        Predict which heavy/light chain pairs are most compatible.
        
        Args:
            heavy_chains: List of heavy chain sequences
            light_chains: List of light chain sequences
            
        Returns:
            Compatibility matrix
        """
        print("Computing pairing compatibility...")
        
        # Get embeddings for all chains
        heavy_embs = [self.get_chain_embedding(hc) for hc in heavy_chains]
        light_embs = [self.get_chain_embedding(lc) for lc in light_chains]
        
        # Compute compatibility (cosine similarity)
        compatibility = np.zeros((len(heavy_chains), len(light_chains)))
        
        for i, h_emb in enumerate(heavy_embs):
            for j, l_emb in enumerate(light_embs):
                similarity = np.dot(h_emb, l_emb) / (
                    np.linalg.norm(h_emb) * np.linalg.norm(l_emb)
                )
                compatibility[i, j] = similarity
        
        return compatibility
    
    def plot_compatibility_matrix(self, compatibility, heavy_ids, light_ids,
                                  output_dir='plots/paired_chains'):
        """
        Visualize chain pairing compatibility.
        
        Args:
            compatibility: Compatibility matrix
            heavy_ids: Heavy chain identifiers
            light_ids: Light chain identifiers
            output_dir: Directory to save plot
        """
        os.makedirs(output_dir, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(
            compatibility,
            xticklabels=light_ids,
            yticklabels=heavy_ids,
            cmap='RdYlGn',
            center=0,
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Compatibility Score'},
            ax=ax
        )
        
        ax.set_title('Heavy/Light Chain Pairing Compatibility', fontsize=14, fontweight='bold')
        ax.set_xlabel('Light Chains')
        ax.set_ylabel('Heavy Chains')
        
        plt.tight_layout()
        
        filename = 'compatibility_matrix.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {filename}")


def create_paired_dataset_example():
    """Create example paired chain dataset."""
    
    # Example paired sequences (shortened for demonstration)
    data = {
        'antibody_id': ['AB001', 'AB002', 'AB003'],
        'heavy_chain': [
            'QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYYMHWVRQAPGQGLEWMGIINPSGGSTSYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYCARGGYDGRGFDYWGQGTLVTVSS',
            'EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDRGYPYFDYWGQGTLVTVSS',
            'QVQLQQSGPGLVKPSQTLSLTCAISGDSVSSNSAAWNWIRQSPSRGLEWLGRTYYRSKWYNDYAVSVKSRITINPDTSKNQFSLQLNSVTPEDTAVYYCARHYDYGDYVWGQGTTVTVSS'
        ],
        'light_chain': [
            'DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK',
            'DIQMTQSPSSLSASVGDRVTITCRASQGISSYLAWYQQKPGKAPKLLIYAASTLQSGVPSRFSGSGSGTEFTLTISSLQPDDFATYYCQQANSFPLTFGGGTKVEIK',
            'EIVLTQSPATLSLSPGERATLSCRASQSVSSYLAWYQQKPGQAPRLLIYDASNRATGIPARFSGSGSGTDFTLTISSLEPEDFAVYYCQQRSNWPLTFGGGTKVEIK'
        ],
        'stability_score': [8.5, 7.9, 9.2],
        'aggregation': [2.1, 2.8, 1.5]
    }
    
    df = pd.DataFrame(data)
    os.makedirs('../../data', exist_ok=True)
    df.to_csv('../../data/paired_chains.csv', index=False)
    print("✅ Created paired_chains.csv")
    
    return df


def main():
    """Main function for paired chain analysis."""
    
    print("\n" + "="*70)
    print("PAIRED HEAVY/LIGHT CHAIN ANALYSIS")
    print("="*70)
    print("\n🧬 Analyzing antibody chains in combination\n")
    
    # Create or load paired dataset
    if not os.path.exists('../../data/paired_chains.csv'):
        df = create_paired_dataset_example()
    else:
        df = pd.read_csv('../../data/paired_chains.csv')
    
    print(f"Loaded {len(df)} paired antibody sequences\n")
    
    # Initialize analyzer
    analyzer = PairedChainAnalyzer()
    
    # Analyze first antibody
    heavy = df.iloc[0]['heavy_chain']
    light = df.iloc[0]['light_chain']
    ab_id = df.iloc[0]['antibody_id']
    
    print(f"Analyzing {ab_id}...")
    analyzer.visualize_chain_interaction(heavy, light, 
                                        output_dir=f'../../plots/paired_chains/{ab_id}')
    
    # Analyze contributions
    contributions = analyzer.analyze_chain_contributions(heavy, light)
    print(f"\nChain similarity: {contributions['heavy_light_similarity']:.3f}")
    
    # Compute compatibility matrix
    print("\nComputing pairing compatibility...")
    compatibility = analyzer.predict_pairing_compatibility(
        df['heavy_chain'].tolist(),
        df['light_chain'].tolist()
    )
    
    analyzer.plot_compatibility_matrix(
        compatibility,
        df['antibody_id'].tolist(),
        df['antibody_id'].tolist(),
        output_dir='../../plots/paired_chains'
    )
    
    print("\n" + "="*70)
    print("✅ PAIRED CHAIN ANALYSIS COMPLETE!")
    print("="*70)
    print("\nPlots saved to: plots/paired_chains/")
    print("\nInsights:")
    print("  • High correlation = chains work well together")
    print("  • Interaction pattern = synergistic effects")
    print("  • Compatibility matrix = optimal pairing predictions")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
