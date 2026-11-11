from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('SouthGermanCredit.asc', sep=r'\s+')

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

    def detect_outliers(self, factor=1.5):

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outliers_summary = {}

        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR

            outliers_count = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)][col].count()

            if outliers_count > 0:
                outliers_summary[col] = outliers_count

        if not outliers_summary:
            print("Outliners not detected!.")
        else:
            print("Outliners detected:")
            for col, count in outliers_summary.items():
                print(f"   - {col}: {count} strings")

    def run_full_analysis(self):
        self.check_structure()
        self.check_missing_and_duplicates()
        self.detect_outliers()

DatasetQualityAnalyzer(df).run_full_analysis()

X = df.drop(columns=[df.columns[-1]]) # відкидаємо цільову змінну для неконтрольованого навчання
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
kmeans_labels = kmeans.fit_predict(X_scaled)

agg_cluster = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
agg_labels = agg_cluster.fit_predict(X_scaled)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmeans_labels, cmap='viridis', s=15)
plt.title("t-SNE + K-Means (2 clusters)")

plt.subplot(1, 2, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=agg_labels, cmap='plasma', s=15)
plt.title("t-SNE + Agglomerative Clustering (2 clusters)")
plt.show()