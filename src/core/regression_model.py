"""
Build regression models to predict antibody developability metrics from embeddings.
This module links learned embeddings to empirical biophysical properties.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os


class DevelopabilityPredictor:
    """Predict antibody developability metrics from sequence embeddings."""
    
    def __init__(self, embedding_type='unirep'):
        """
        Initialize predictor.
        
        Args:
            embedding_type: Type of embeddings to use ('unirep', 'protbert', or 'combined')
        """
        self.embedding_type = embedding_type
        self.models = {}
        self.scalers = {}
        self.metrics = ['solubility', 'aggregation_propensity', 'stability_score', 
                       'tm_celsius', 'expression_yield']
    
    def prepare_data(self, embeddings_path, sequences_csv):
        """
        Load and prepare data for training.
        
        Args:
            embeddings_path: Path to embeddings .npz file
            sequences_csv: Path to CSV with sequences and labels
            
        Returns:
            X: Feature matrix
            y_dict: Dictionary of target variables
            df: DataFrame with metadata
        """
        # Load embeddings
        print(f"Loading embeddings from {embeddings_path}...")
        emb_data = np.load(embeddings_path, allow_pickle=True)
        
        # Load labels
        df = pd.read_csv(sequences_csv)
        
        # Extract embeddings based on type
        if self.embedding_type == 'unirep':
            X = emb_data['unirep']
        elif self.embedding_type == 'protbert':
            X = emb_data['protbert']
        elif self.embedding_type == 'combined':
            X = np.concatenate([emb_data['unirep'], emb_data['protbert']], axis=1)
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")
        
        print(f"Feature matrix shape: {X.shape}")
        
        # Extract target variables
        y_dict = {metric: df[metric].values for metric in self.metrics}
        
        return X, y_dict, df
    
    def train_model(self, X_train, y_train, model_type='random_forest'):
        """
        Train a single regression model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            model_type: Type of model to train
            
        Returns:
            Trained model
        """
        if model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == 'ridge':
            model = Ridge(alpha=1.0)
        elif model_type == 'svr':
            model = SVR(kernel='rbf', C=1.0, gamma='scale')
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = model.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        return metrics, y_pred
    
    def train_all_metrics(self, X, y_dict, test_size=0.2, model_type='random_forest'):
        """
        Train models for all developability metrics.
        
        Args:
            X: Feature matrix
            y_dict: Dictionary of target variables
            test_size: Fraction of data for testing
            model_type: Type of model to train
            
        Returns:
            Dictionary of results
        """
        results = {}
        
        # Auto-adjust for small datasets
        n_samples = X.shape[0]
        original_model_type = model_type
        
        if n_samples < 50:
            print("\n" + "⚠️ "*30)
            print(f"WARNING: Small dataset detected ({n_samples} sequences)")
            print(f"Minimum recommended: 50 sequences")
            print(f"Optimal: 100-500 sequences")
            print(f"\nWith {n_samples} sequences:")
            print(f"  → Switching from {model_type} to 'ridge' (simpler model)")
            print(f"  → Using cross-validation for better estimates")
            print(f"  → Predictions may be unreliable - use for exploration only")
            print("⚠️ "*30 + "\n")
            model_type = 'ridge'
            
            if n_samples < 30:
                print("💡 TIP: Focus on clustering & visualization rather than prediction")
                print("   PCA, t-SNE, and UMAP are more reliable with small datasets\n")
        
        for metric in self.metrics:
            print(f"\n{'='*60}")
            print(f"Training model for: {metric}")
            print(f"{'='*60}")
            
            y = y_dict[metric]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = self.train_model(X_train_scaled, y_train, model_type)
            
            # Evaluate
            eval_metrics, y_pred = self.evaluate_model(model, X_test_scaled, y_test)
            
            # Cross-validation score
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train, cv=5, 
                scoring='r2', n_jobs=-1
            )
            
            # Store results
            results[metric] = {
                'model': model,
                'scaler': scaler,
                'metrics': eval_metrics,
                'cv_scores': cv_scores,
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            # Print results
            print(f"\nTest Set Performance:")
            print(f"  R² Score: {eval_metrics['r2']:.4f}")
            print(f"  RMSE: {eval_metrics['rmse']:.4f}")
            print(f"  MAE: {eval_metrics['mae']:.4f}")
            print(f"\nCross-Validation R² Scores:")
            print(f"  Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Store trained models
            self.models[metric] = model
            self.scalers[metric] = scaler
        
        return results
    
    def predict(self, X, metric):
        """
        Make predictions for a specific metric.
        
        Args:
            X: Feature matrix
            metric: Metric to predict
            
        Returns:
            Predictions
        """
        if metric not in self.models:
            raise ValueError(f"No trained model for metric: {metric}")
        
        X_scaled = self.scalers[metric].transform(X)
        return self.models[metric].predict(X_scaled)
    
    def save_models(self, output_dir):
        """
        Save trained models to disk.
        
        Args:
            output_dir: Directory to save models
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for metric in self.metrics:
            if metric in self.models:
                model_path = os.path.join(output_dir, f'{metric}_{self.embedding_type}_model.pkl')
                scaler_path = os.path.join(output_dir, f'{metric}_{self.embedding_type}_scaler.pkl')
                
                joblib.dump(self.models[metric], model_path)
                joblib.dump(self.scalers[metric], scaler_path)
                print(f"Saved model for {metric}")
    
    def load_models(self, input_dir):
        """
        Load trained models from disk.
        
        Args:
            input_dir: Directory containing saved models
        """
        for metric in self.metrics:
            model_path = os.path.join(input_dir, f'{metric}_{self.embedding_type}_model.pkl')
            scaler_path = os.path.join(input_dir, f'{metric}_{self.embedding_type}_scaler.pkl')
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.models[metric] = joblib.load(model_path)
                self.scalers[metric] = joblib.load(scaler_path)
                print(f"Loaded model for {metric}")
    
    def plot_results(self, results, output_dir='plots'):
        """
        Generate visualization plots for model results.
        
        Args:
            results: Dictionary of results from train_all_metrics
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
        
        # 1. Performance comparison across metrics
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Model Performance - {self.embedding_type.upper()} Embeddings', 
                     fontsize=16, fontweight='bold')
        
        metrics_names = list(results.keys())
        r2_scores = [results[m]['metrics']['r2'] for m in metrics_names]
        rmse_scores = [results[m]['metrics']['rmse'] for m in metrics_names]
        mae_scores = [results[m]['metrics']['mae'] for m in metrics_names]
        cv_means = [results[m]['cv_scores'].mean() for m in metrics_names]
        
        # R² scores
        axes[0, 0].bar(range(len(metrics_names)), r2_scores, color='steelblue', alpha=0.7)
        axes[0, 0].set_xticks(range(len(metrics_names)))
        axes[0, 0].set_xticklabels(metrics_names, rotation=45, ha='right')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_title('Test Set R² Scores')
        axes[0, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='0.5 threshold')
        axes[0, 0].legend()
        
        # RMSE
        axes[0, 1].bar(range(len(metrics_names)), rmse_scores, color='coral', alpha=0.7)
        axes[0, 1].set_xticks(range(len(metrics_names)))
        axes[0, 1].set_xticklabels(metrics_names, rotation=45, ha='right')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title('Root Mean Squared Error')
        
        # MAE
        axes[1, 0].bar(range(len(metrics_names)), mae_scores, color='mediumseagreen', alpha=0.7)
        axes[1, 0].set_xticks(range(len(metrics_names)))
        axes[1, 0].set_xticklabels(metrics_names, rotation=45, ha='right')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].set_title('Mean Absolute Error')
        
        # CV R² scores
        axes[1, 1].bar(range(len(metrics_names)), cv_means, color='mediumpurple', alpha=0.7)
        axes[1, 1].set_xticks(range(len(metrics_names)))
        axes[1, 1].set_xticklabels(metrics_names, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Mean CV R²')
        axes[1, 1].set_title('Cross-Validation R² Scores')
        axes[1, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'performance_summary_{self.embedding_type}.png'), 
                    bbox_inches='tight')
        plt.close()
        
        # 2. Prediction vs Actual plots for each metric
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_names):
            y_test = results[metric]['y_test']
            y_pred = results[metric]['y_pred']
            r2 = results[metric]['metrics']['r2']
            
            axes[idx].scatter(y_test, y_pred, alpha=0.6, s=50)
            axes[idx].plot([y_test.min(), y_test.max()], 
                          [y_test.min(), y_test.max()], 
                          'r--', lw=2, label='Perfect prediction')
            axes[idx].set_xlabel(f'Actual {metric}')
            axes[idx].set_ylabel(f'Predicted {metric}')
            axes[idx].set_title(f'{metric}\nR² = {r2:.3f}')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        # Hide extra subplot
        if len(metrics_names) < len(axes):
            axes[-1].axis('off')
        
        fig.suptitle(f'Predicted vs Actual Values - {self.embedding_type.upper()}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'predictions_scatter_{self.embedding_type}.png'), 
                    bbox_inches='tight')
        plt.close()
        
        print(f"\nPlots saved to {output_dir}/")


def compare_embedding_types(embeddings_path, sequences_csv, model_type='random_forest'):
    """
    Compare performance of different embedding types.
    
    Args:
        embeddings_path: Path to embeddings file
        sequences_csv: Path to sequences CSV
        model_type: Type of model to use
    """
    # Check which embeddings are available
    emb_data = np.load(embeddings_path, allow_pickle=True)
    available_embeddings = []
    
    if 'unirep' in emb_data.files:
        available_embeddings.append('unirep')
    if 'protbert' in emb_data.files:
        available_embeddings.append('protbert')
    if 'unirep' in emb_data.files and 'protbert' in emb_data.files:
        available_embeddings.append('combined')
    
    if not available_embeddings:
        print("❌ Error: No embeddings found in file!")
        return {}
    
    print(f"\n📊 Available embeddings: {', '.join(available_embeddings)}")
    comparison_results = {}
    
    for emb_type in available_embeddings:
        print(f"\n{'#'*70}")
        print(f"# Evaluating {emb_type.upper()} embeddings")
        print(f"{'#'*70}")
        
        predictor = DevelopabilityPredictor(embedding_type=emb_type)
        X, y_dict, df = predictor.prepare_data(embeddings_path, sequences_csv)
        results = predictor.train_all_metrics(X, y_dict, model_type=model_type)
        
        # Save models
        predictor.save_models(f'../models/{emb_type}')
        
        # Generate plots
        predictor.plot_results(results, output_dir=f'../plots/{emb_type}')
        
        comparison_results[emb_type] = results
    
    # Create comparison summary
    if comparison_results:
        print("\n" + "="*70)
        print("EMBEDDING TYPE COMPARISON SUMMARY")
        print("="*70)
        
        for metric in predictor.metrics:
            print(f"\n{metric.upper()}:")
            print(f"  {'Embedding Type':<15} {'R² Score':<12} {'RMSE':<12} {'CV R² Mean':<12}")
            print(f"  {'-'*60}")
            for emb_type in available_embeddings:
                r2 = comparison_results[emb_type][metric]['metrics']['r2']
                rmse = comparison_results[emb_type][metric]['metrics']['rmse']
                cv_mean = comparison_results[emb_type][metric]['cv_scores'].mean()
                print(f"  {emb_type:<15} {r2:<12.4f} {rmse:<12.4f} {cv_mean:<12.4f}")
    
    return comparison_results


def main():
    """Main function to train and evaluate regression models."""
    
    # Paths
    embeddings_path = '../../data/embeddings.npz'
    sequences_csv = '../../datasets/example_sequences.csv'
    
    # Compare all embedding types
    compare_embedding_types(embeddings_path, sequences_csv, model_type='random_forest')
    
    print("\n" + "="*70)
    print("REGRESSION MODELING COMPLETE")
    print("="*70)
    print("Models saved to: models/")
    print("Plots saved to: plots/")


if __name__ == "__main__":
    main()
