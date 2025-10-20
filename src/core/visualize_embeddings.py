"""
Visualize protein sequence embeddings using dimensionality reduction techniques.
Identify clusters of high-stability proteins and link to biophysical properties.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


class EmbeddingVisualizer:
    """Visualize and analyze protein sequence embeddings."""
    
    def __init__(self, embeddings_path, sequences_csv):
        """
        Initialize visualizer with embeddings and metadata.
        
        Args:
            embeddings_path: Path to embeddings .npz file
            sequences_csv: Path to CSV with sequences and labels
        """
        # Load embeddings
        print("Loading embeddings...")
        emb_data = np.load(embeddings_path, allow_pickle=True)
        
        # Load available embeddings
        self.unirep = emb_data['unirep'] if 'unirep' in emb_data.files else None
        self.protbert = emb_data['protbert'] if 'protbert' in emb_data.files else None
        
        # Create combined if both available
        if self.unirep is not None and self.protbert is not None:
            self.combined = np.concatenate([self.unirep, self.protbert], axis=1)
        else:
            self.combined = None
        
        # Load metadata
        self.df = pd.read_csv(sequences_csv)
        print(f"Loaded {len(self.df)} sequences")
        
        # Print available embeddings
        available = []
        if self.unirep is not None:
            available.append('UniRep')
        if self.protbert is not None:
            available.append('ProtBERT')
        if self.combined is not None:
            available.append('Combined')
        print(f"📊 Available embeddings: {', '.join(available)}")
        
        # Store reduced representations
        self.reduced_embeddings = {}
    
    def reduce_dimensions_pca(self, X, n_components=2):
        """
        Reduce dimensions using PCA.
        
        Args:
            X: High-dimensional embeddings
            n_components: Number of principal components
            
        Returns:
            Reduced embeddings, explained variance
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X_scaled)
        
        explained_var = pca.explained_variance_ratio_
        
        print(f"PCA: {n_components} components explain "
              f"{explained_var.sum()*100:.2f}% of variance")
        
        return X_reduced, explained_var, pca
    
    def reduce_dimensions_tsne(self, X, n_components=2, perplexity=30):
        """
        Reduce dimensions using t-SNE.
        
        Args:
            X: High-dimensional embeddings
            n_components: Number of components
            perplexity: t-SNE perplexity parameter
            
        Returns:
            Reduced embeddings
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # For large feature spaces, first reduce with PCA
        # Use min(50, n_samples-1) to avoid errors with small datasets
        if X.shape[1] > 50:
            n_components_pca = min(50, X.shape[0] - 1)
            pca = PCA(n_components=n_components_pca)
            X_scaled = pca.fit_transform(X_scaled)
            print(f"Pre-reduced to {n_components_pca} dims with PCA (explains "
                  f"{pca.explained_variance_ratio_.sum()*100:.2f}% variance)")
        
        # Adjust perplexity for small datasets
        # t-SNE requires perplexity < n_samples
        perplexity = min(perplexity, max(5, X.shape[0] - 1))
        
        tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                    random_state=42, n_iter=1000)
        X_reduced = tsne.fit_transform(X_scaled)
        
        print("t-SNE reduction complete")
        
        return X_reduced
    
    def reduce_dimensions_umap(self, X, n_components=2, n_neighbors=15):
        """
        Reduce dimensions using UMAP.
        
        Args:
            X: High-dimensional embeddings
            n_components: Number of components
            n_neighbors: UMAP n_neighbors parameter
            
        Returns:
            Reduced embeddings
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Adjust n_neighbors for small datasets
        # UMAP requires n_neighbors < n_samples
        n_neighbors = min(n_neighbors, X.shape[0] - 1)
        
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                           random_state=42, min_dist=0.1)
        X_reduced = reducer.fit_transform(X_scaled)
        
        print("UMAP reduction complete")
        
        return X_reduced
    
    def identify_clusters(self, X_reduced, method='kmeans', n_clusters=3):
        """
        Identify clusters in reduced embedding space.
        
        Args:
            X_reduced: Reduced embeddings
            method: Clustering method ('kmeans' or 'dbscan')
            n_clusters: Number of clusters (for kmeans)
            
        Returns:
            Cluster labels
        """
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clusterer.fit_predict(X_reduced)
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=3)
            labels = clusterer.fit_predict(X_reduced)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        print(f"Identified {len(np.unique(labels))} clusters using {method}")
        return labels
    
    def plot_2d_embeddings(self, embedding_type='unirep', method='pca', 
                          color_by='stability_score', output_dir='plots'):
        """
        Create 2D visualization of embeddings.
        
        Args:
            embedding_type: Type of embeddings ('unirep', 'protbert', 'combined')
            method: Dimensionality reduction method ('pca', 'tsne', 'umap')
            color_by: Metadata column to color points by
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Select embeddings
        if embedding_type == 'unirep':
            X = self.unirep
        elif embedding_type == 'protbert':
            X = self.protbert
        elif embedding_type == 'combined':
            X = self.combined
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
        
        # Check if embeddings are available
        if X is None:
            print(f"⚠️  Skipping {embedding_type} - embeddings not available")
            return
        
        # Reduce dimensions
        print(f"\nReducing {embedding_type} embeddings using {method.upper()}...")
        if method == 'pca':
            X_reduced, explained_var, pca_obj = self.reduce_dimensions_pca(X)
            method_title = f"PCA ({explained_var[0]*100:.1f}%, {explained_var[1]*100:.1f}%)"
        elif method == 'tsne':
            X_reduced = self.reduce_dimensions_tsne(X)
            method_title = "t-SNE"
        elif method == 'umap':
            X_reduced = self.reduce_dimensions_umap(X)
            method_title = "UMAP"
        else:
            raise ValueError(f"Unknown reduction method: {method}")
        
        # Store for later use
        key = f"{embedding_type}_{method}"
        self.reduced_embeddings[key] = X_reduced
        
        # Identify clusters
        cluster_labels = self.identify_clusters(X_reduced, method='kmeans', n_clusters=3)
        
        # Create matplotlib figure
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'{embedding_type.upper()} Embeddings - {method_title}', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Colored by stability metric
        scatter1 = axes[0].scatter(X_reduced[:, 0], X_reduced[:, 1], 
                                   c=self.df[color_by], cmap='viridis',
                                   s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        axes[0].set_xlabel(f'{method.upper()} 1')
        axes[0].set_ylabel(f'{method.upper()} 2')
        axes[0].set_title(f'Colored by {color_by}')
        plt.colorbar(scatter1, ax=axes[0])
        
        # Plot 2: Colored by clusters
        scatter2 = axes[1].scatter(X_reduced[:, 0], X_reduced[:, 1], 
                                   c=cluster_labels, cmap='tab10',
                                   s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        axes[1].set_xlabel(f'{method.upper()} 1')
        axes[1].set_ylabel(f'{method.upper()} 2')
        axes[1].set_title(f'K-means Clusters (k=3)')
        plt.colorbar(scatter2, ax=axes[1])
        
        # Plot 3: High-stability highlighting
        high_stability = self.df[color_by] > self.df[color_by].quantile(0.75)
        colors = ['red' if hs else 'lightgray' for hs in high_stability]
        axes[2].scatter(X_reduced[:, 0], X_reduced[:, 1], 
                       c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        axes[2].set_xlabel(f'{method.upper()} 1')
        axes[2].set_ylabel(f'{method.upper()} 2')
        axes[2].set_title(f'High-{color_by} Proteins (top 25%)')
        
        plt.tight_layout()
        filename = f'{embedding_type}_{method}_2d.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
        
        # Create interactive Plotly figure
        self._create_interactive_plot(X_reduced, embedding_type, method, 
                                     color_by, cluster_labels, output_dir)
        
        # Analyze clusters
        self._analyze_clusters(cluster_labels, color_by)
    
    def _create_interactive_plot(self, X_reduced, embedding_type, method, 
                                color_by, cluster_labels, output_dir):
        """Create interactive Plotly visualization."""
        
        df_plot = pd.DataFrame({
            f'{method}_1': X_reduced[:, 0],
            f'{method}_2': X_reduced[:, 1],
            'sequence_id': self.df['sequence_id'],
            color_by: self.df[color_by],
            'cluster': cluster_labels,
            'solubility': self.df['solubility'],
            'aggregation': self.df['aggregation_propensity'],
            'stability': self.df['stability_score'],
            'tm': self.df['tm_celsius'],
            'expression': self.df['expression_yield']
        })
        
        fig = px.scatter(
            df_plot, 
            x=f'{method}_1', 
            y=f'{method}_2',
            color=color_by,
            hover_data=['sequence_id', 'solubility', 'aggregation', 
                       'stability', 'tm', 'expression'],
            title=f'{embedding_type.upper()} Embeddings - {method.upper()}',
            color_continuous_scale='viridis',
            width=900,
            height=700
        )
        
        fig.update_traces(marker=dict(size=12, line=dict(width=1, color='black')))
        fig.update_layout(font=dict(size=12))
        
        filename = f'{embedding_type}_{method}_interactive.html'
        fig.write_html(os.path.join(output_dir, filename))
        print(f"Saved interactive plot: {filename}")
    
    def _analyze_clusters(self, cluster_labels, metric='stability_score'):
        """Analyze cluster characteristics."""
        
        df_analysis = self.df.copy()
        df_analysis['cluster'] = cluster_labels
        
        print(f"\n{'='*60}")
        print(f"Cluster Analysis - {metric}")
        print(f"{'='*60}")
        
        for cluster_id in np.unique(cluster_labels):
            cluster_data = df_analysis[df_analysis['cluster'] == cluster_id]
            print(f"\nCluster {cluster_id} (n={len(cluster_data)}):")
            print(f"  Solubility:        {cluster_data['solubility'].mean():.2f} ± {cluster_data['solubility'].std():.2f}")
            print(f"  Aggregation:       {cluster_data['aggregation_propensity'].mean():.2f} ± {cluster_data['aggregation_propensity'].std():.2f}")
            print(f"  Stability Score:   {cluster_data['stability_score'].mean():.2f} ± {cluster_data['stability_score'].std():.2f}")
            print(f"  Tm (°C):           {cluster_data['tm_celsius'].mean():.2f} ± {cluster_data['tm_celsius'].std():.2f}")
            print(f"  Expression Yield:  {cluster_data['expression_yield'].mean():.2f} ± {cluster_data['expression_yield'].std():.2f}")
    
    def plot_pca_variance(self, embedding_type='unirep', n_components=20, output_dir='plots'):
        """
        Plot explained variance for PCA components.
        
        Args:
            embedding_type: Type of embeddings
            n_components: Number of components to analyze
            output_dir: Directory to save plot
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if embedding_type == 'unirep':
            X = self.unirep
        elif embedding_type == 'protbert':
            X = self.protbert
        elif embedding_type == 'combined':
            X = self.combined
        
        # Check if embeddings are available
        if X is None:
            print(f"⚠️  Skipping PCA variance for {embedding_type} - embeddings not available")
            return
        
        _, explained_var, _ = self.reduce_dimensions_pca(X, n_components=n_components)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Individual variance
        ax1.bar(range(1, len(explained_var) + 1), explained_var, alpha=0.7, color='steelblue')
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title(f'Variance Explained by Each Component - {embedding_type.upper()}')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative variance
        cumulative_var = np.cumsum(explained_var)
        ax2.plot(range(1, len(cumulative_var) + 1), cumulative_var, 
                marker='o', color='coral', linewidth=2)
        ax2.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% threshold')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title(f'Cumulative Variance - {embedding_type.upper()}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'{embedding_type}_pca_variance.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
    
    def create_comprehensive_dashboard(self, output_dir='plots'):
        """Create comprehensive visualization dashboard."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        embedding_types = ['unirep', 'protbert', 'combined']
        methods = ['pca', 'tsne', 'umap']
        
        print("\n" + "="*70)
        print("GENERATING COMPREHENSIVE VISUALIZATION DASHBOARD")
        print("="*70)
        
        for emb_type in embedding_types:
            print(f"\n--- {emb_type.upper()} embeddings ---")
            
            # PCA variance analysis
            self.plot_pca_variance(emb_type, n_components=20, output_dir=output_dir)
            
            # Create visualizations for different metrics
            metrics = ['stability_score', 'solubility', 'tm_celsius']
            
            for method in methods:
                for metric in metrics:
                    self.plot_2d_embeddings(
                        embedding_type=emb_type,
                        method=method,
                        color_by=metric,
                        output_dir=output_dir
                    )
        
        print(f"\n{'='*70}")
        print("VISUALIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"All plots saved to: {output_dir}/")


def main():
    """Main function to generate visualizations."""
    
    # Paths
    embeddings_path = '../../data/embeddings.npz'
    sequences_csv = '../../datasets/example_sequences.csv'
    
    # Create visualizer
    visualizer = EmbeddingVisualizer(embeddings_path, sequences_csv)
    
    # Generate comprehensive dashboard
    visualizer.create_comprehensive_dashboard(output_dir='../../plots/visualizations')
    
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print("✓ PCA captures linear variance in embedding space")
    print("✓ t-SNE reveals local neighborhood structure")
    print("✓ UMAP balances global and local structure preservation")
    print("✓ Clusters identify groups of proteins with similar properties")
    print("✓ High-stability proteins can be identified in latent space")


if __name__ == "__main__":
    main()
