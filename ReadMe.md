# South German Credit Data Clustering & Dimensionality Reduction

This repository contains a Python script for exploratory data analysis, data quality checks, unsupervised clustering, and dimensionality reduction using the **South German Credit** dataset.

## 📋 Overview of the Pipeline

1. **Dataset Quality Analysis (`DatasetQualityAnalyzer`)**:
   * Inspects dataset dimensions (number of rows and columns).
   * Verifies data types across all features.
   * Checks for duplicate records and missing values ($\text{NaN}$s).

2. **Feature Scaling**:
   * Drops the target column and normalizes features using `StandardScaler` to ensure uniform scale contribution during distance-based modeling.

3. **Dimensionality Reduction (`t-SNE`)**:
   * Projects high-dimensional feature spaces down to 2 dimensions using t-SNE (`perplexity=30`, `random_state=42`) for visual inspection.

4. **Clustering Algorithms**:
   * **K-Means Clustering**: Unsupervised partitioning into 2 clusters using centroid-based minimization.
   * **Agglomerative Clustering**: Hierarchical bottom-up clustering using Ward linkage and Euclidean distance.

5. **Visualization & Comparison**:
   * Generates side-by-side scatter plots comparing 2D t-SNE projections colored by cluster assignments for both algorithms.

---

## 🛠️ Code Implementation

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
 
df = pd.read_csv('SouthGermanCredit.asc', sep=r'\s+')

# Аналізуємо датасет на пропуски, дуплікати і тд. :
class DatasetQualityAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def check_structure(self):
        print(f"String number: {self.df.shape[0]}")
        print(f"Collumns number: {self.df.shape[1]}\n")
        print("Datatypes:")
        print(self.df.dtypes.to_string())
        print("\n")

    def check_missing_and_duplicates(self):
        duplicates = self.df.duplicated().sum()
        print(f"Duplicates: {duplicates}")

        missing = self.df.isnull().sum()
        missing_filtered = missing[missing > 0]

        if missing_filtered.empty:
            print("NaNs are missing from dataset! 👍")
        else:
            print("Number of missing values per column:")
            print(missing_filtered.to_string())
        print("\n")

    def run_full_analysis(self):
        self.check_structure()
        self.check_missing_and_duplicates()

DatasetQualityAnalyzer(df).run_full_analysis()

# Масштабування ознак
X = df.drop(columns=[df.columns[-1]])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Двомірна репрезентація за допомогою t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Тренування моделі 1: K-Means
kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
kmeans_labels = kmeans.fit_predict(X_scaled)

# Тренування моделі 2: Agglomerative Clustering
agg_cluster = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
agg_labels = agg_cluster.fit_predict(X_scaled)

# Візуалізація результатів t-SNE та кластеризації
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmeans_labels, cmap='viridis', s=15)
plt.title("t-SNE + K-Means (2 clusters)")

plt.subplot(1, 2, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=agg_labels, cmap='plasma', s=15)
plt.title("t-SNE + Agglomerative Clustering (2 clusters)")
plt.show()