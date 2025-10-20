# Understanding R² Scores

## 📊 What is R²?

**R² (R-squared)** or **Coefficient of Determination** measures how well your model's predictions match the actual data.

### Mathematical Definition

```
R² = 1 - (Sum of Squared Errors / Total Variance)

R² = 1 - (Σ(y_true - y_pred)² / Σ(y_true - y_mean)²)
```

### Interpretation

| R² Range | Interpretation | What It Means |
|----------|----------------|---------------|
| **1.00** | Perfect | Model predicts every value exactly |
| **0.80-0.99** | Excellent | Model explains 80-99% of variance |
| **0.60-0.79** | Good | Useful for screening, good predictions |
| **0.40-0.59** | Moderate | Some predictive power, use with caution |
| **0.20-0.39** | Weak | Poor predictions, barely better than mean |
| **0.00-0.19** | Very Weak | Almost no predictive power |
| **< 0.00** | **NEGATIVE** | **Worse than predicting the mean!** |

---

## 🚨 Negative R² - What Went Wrong?

### What Does Negative R² Mean?

**Your model's predictions are WORSE than just guessing the average every time.**

### Example

```python
# Your actual stability scores
True values:  [8.5, 7.9, 8.2, 7.7, 8.1]
Mean:         8.08

# Baseline (always predict mean)
Baseline:     [8.08, 8.08, 8.08, 8.08, 8.08]
Error:        Small
R² = 0.00     ← Baseline performance

# Your model's predictions (random/bad)
Model:        [6.5, 9.2, 5.8, 9.5, 6.2]
Error:        Large!
R² = -1.25    ← WORSE than baseline!
```

### Why This Happens

1. **Insufficient Training Data**
   - Need: 50-100+ samples minimum
   - You have: 20 samples (only 16 for training!)
   
2. **Overfitting**
   - Model memorizes training data
   - Fails completely on test data
   
3. **High Dimensionality**
   - Features: 1024 (ProtBERT) or 1280 (ESM-2)
   - Samples: 20
   - Ratio: 50:1 features per sample (should be 1:10!)
   
4. **Random Noise**
   - With so little data, model learns noise instead of patterns

---

## ✅ Solutions

### Solution 1: Get More Data (Best)

```
Current:        20 sequences  → R² = -0.6 (bad)
Minimum:        50 sequences  → R² = 0.4-0.6 (okay)
Recommended:   100 sequences  → R² = 0.6-0.75 (good)
Optimal:       200+ sequences → R² = 0.75-0.85 (excellent)
```

**How to get more data:**
- Collect more experimental measurements
- Use public databases (PDB, SAbDab, etc.)
- Combine datasets from related studies
- Run more experiments on similar proteins

### Solution 2: Use Simpler Models

For small datasets (< 50 samples), use **regularized linear models**:

```python
from sklearn.linear_model import Ridge, Lasso

# Ridge regression (L2 regularization)
model = Ridge(alpha=1.0)

# Or Lasso (L1 regularization, feature selection)
model = Lasso(alpha=0.1)
```

**Why this helps:**
- Fewer parameters to learn
- Less prone to overfitting
- More stable with limited data

### Solution 3: Dimensionality Reduction First

Reduce features before modeling:

```python
from sklearn.decomposition import PCA

# Reduce 1024 features to 10-20 components
pca = PCA(n_components=min(20, n_samples - 1))
X_reduced = pca.fit_transform(X)

# Then train model
model.fit(X_reduced, y)
```

### Solution 4: Cross-Validation for Small Datasets

Use **Leave-One-Out (LOO)** cross-validation:

```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = []

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    scores.append(score)

print(f"LOO R²: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

### Solution 5: Focus on Exploration, Not Prediction

**With 20 sequences, focus on:**

✅ **Clustering**
```python
from sklearn.cluster import KMeans

# Find groups of similar antibodies
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(embeddings)
```

✅ **Visualization**
```python
# PCA, t-SNE, UMAP
# See which antibodies are similar
# Identify outliers
```

✅ **Similarity Search**
```python
# Find antibodies similar to a query
from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(query_embedding, all_embeddings)
most_similar = np.argsort(similarities)[-5:]  # Top 5
```

❌ **NOT Prediction** (too unreliable with 20 samples)

---

## 📈 Expected R² by Dataset Size

### Protein Property Prediction

| Dataset Size | Expected R² | Reliability |
|--------------|-------------|-------------|
| 20 sequences | -1.0 to 0.3 | Unreliable ❌ |
| 50 sequences | 0.3 to 0.5 | Poor ⚠️ |
| 100 sequences | 0.5 to 0.65 | Moderate ✓ |
| 200 sequences | 0.65 to 0.75 | Good ✓✓ |
| 500 sequences | 0.75 to 0.85 | Excellent ✓✓✓ |
| 1000+ sequences | 0.80 to 0.90 | Outstanding ✓✓✓✓ |

*Assumes good quality labels and appropriate model*

---

## 🔍 Diagnosing Your Results

### Your Current Results (20 sequences)

```
SOLUBILITY:           R² = -0.6271  ← Worse than guessing mean
AGGREGATION:          R² = -0.6048  ← Worse than guessing mean
STABILITY:            R² = -1.1559  ← Much worse than mean
TM_CELSIUS:           R² = -0.8652  ← Worse than guessing mean
EXPRESSION_YIELD:     R² = -0.5937  ← Worse than guessing mean
```

**Diagnosis**: All metrics show negative R², indicating:
1. ❌ Too few samples (20 vs recommended 100+)
2. ❌ Model overfitting to training data
3. ❌ Predictions are unreliable

**Recommendation**: 
- Collect 50-100+ more sequences for reliable predictions
- Use current data for clustering/visualization only
- Switch to Ridge regression (automatic with updated code)

---

## 💡 Practical Guidelines

### When to Trust Your Model

✅ **Trust predictions when:**
- R² > 0.70 on test set
- R² > 0.60 on cross-validation
- Dataset has 100+ samples
- Training and test R² are similar (not overfitting)

❌ **Don't trust predictions when:**
- R² < 0.40 (too unreliable)
- R² is negative (worse than baseline)
- Large gap between training and test R² (overfitting)
- Dataset has < 50 samples

### What to Report

Always report **both**:
1. **Test set R²** (generalization performance)
2. **Cross-validation R²** (more robust estimate)

Example:
```
Model Performance:
  Test R²: 0.72
  CV R² (5-fold): 0.68 ± 0.12
  
Interpretation: Good predictive power, reliable for screening
```

---

## 📚 References

1. **R² Interpretation**
   - R² = 1 - RSS/TSS where RSS = residual sum of squares, TSS = total sum of squares
   - Negative R² means RSS > TSS (model worse than mean)

2. **Small Sample Guidelines**
   - Recommended: 10-20 samples per feature (Harrell, 2015)
   - Your ratio: 1024 features / 20 samples = 51 features/sample ❌

3. **Protein ML Best Practices**
   - Minimum 50-100 sequences for basic models
   - 200+ for robust predictions
   - 1000+ for deep learning

---

## 🎯 Next Steps for You

### Short-term (Current 20 sequences)

1. ✅ Use **clustering** to find similar antibodies
2. ✅ Use **visualization** (PCA, t-SNE, UMAP) to explore data
3. ✅ Use **similarity search** to find nearest neighbors
4. ❌ Don't rely on predictions (R² too low)

### Long-term (Collect more data)

1. **Target: 100 sequences**
   - Will achieve R² = 0.6-0.7 (usable predictions)
   - Can screen new candidates reliably
   
2. **Target: 200+ sequences**
   - Will achieve R² = 0.7-0.8 (good predictions)
   - Can use for rational design
   
3. **Use ESM-2 embeddings**
   - 5-15% better than ProtBERT
   - Already running in your advanced features script!

---

**Bottom Line**: With 20 sequences, focus on **exploration** (clustering, visualization) rather than **prediction**. Collect more data for reliable models!
