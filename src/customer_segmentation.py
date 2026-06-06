"""
Customer Segmentation using K-Means Clustering
================================================
Unsupervised ML project to identify customer segments.

Author: Rolivhuwa Thomoli
Date: June 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")


def generate_customer_data(n_samples=5000, random_state=42):
    """Generate realistic customer dataset."""
    np.random.seed(random_state)

    data = {
        'age': np.random.randint(18, 71, n_samples),
        'annual_income': np.random.lognormal(10.8, 0.6, n_samples).astype(int),
        'spending_score': np.random.randint(1, 101, n_samples),
        'membership_years': np.random.exponential(3, n_samples).clip(0, 15).round(1),
        'purchase_frequency': np.random.poisson(12, n_samples).clip(1, 50),
        'last_purchase_amount': np.random.lognormal(4, 1.2, n_samples).round(2),
    }

    df = pd.DataFrame(data)
    df['annual_income'] = df['annual_income'].clip(15000, 200000)
    df['last_purchase_amount'] = df['last_purchase_amount'].clip(10, 2000)
    df['membership_years'] = df['membership_years'].clip(0, 15)

    return df


def find_optimal_clusters(X, max_k=10):
    """Use Elbow Method and Silhouette Score to find optimal k."""
    print("\n🔍 Finding Optimal Number of Clusters...")

    inertias = []
    silhouette_scores = []
    K_range = range(2, max_k + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))

    # Plot Elbow Method
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Inertia (WCSS)')
    axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
    axes[0].axvline(x=4, color='r', linestyle='--', alpha=0.7, label='Optimal k=4')
    axes[0].legend()

    axes[1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
    axes[1].axvline(x=4, color='r', linestyle='--', alpha=0.7, label='Optimal k=4')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('../images/elbow_method.png', dpi=150, bbox_inches='tight')
    print("✅ Elbow method plot saved")
    plt.close()

    best_k = list(K_range)[np.argmax(silhouette_scores)]
    print(f"Best Silhouette Score: {max(silhouette_scores):.3f} at k={best_k}")
    return best_k


def apply_clustering(df, X_scaled, n_clusters=4):
    """Apply K-Means and profile clusters."""
    print(f"\n🎯 Applying K-Means with k={n_clusters}...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    print(f"Silhouette Score: {silhouette_score(X_scaled, df['cluster']):.3f}")

    # Cluster profiles
    print("\n📊 Cluster Profiles:")
    print("-" * 60)
    cluster_summary = df.groupby('cluster').agg({
        'age': 'mean',
        'annual_income': 'mean',
        'spending_score': 'mean',
        'membership_years': 'mean',
        'purchase_frequency': 'mean',
        'last_purchase_amount': 'mean'
    }).round(1)
    print(cluster_summary.to_string())

    # Cluster sizes
    print("\n📈 Cluster Sizes:")
    print(df['cluster'].value_counts().sort_index().to_string())

    return df, kmeans


def visualize_clusters(df, X_scaled):
    """Create cluster visualizations."""
    # PCA for 2D visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['pca1'] = X_pca[:, 0]
    df['pca2'] = X_pca[:, 1]

    # 2D scatter plot
    plt.figure(figsize=(10, 8))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    for i, color in enumerate(colors):
        cluster_data = df[df['cluster'] == i]
        plt.scatter(cluster_data['pca1'], cluster_data['pca2'],
                    c=color, label=f'Cluster {i}', alpha=0.6, s=30)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('Customer Segments (PCA Projection)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../images/clusters_2d.png', dpi=150, bbox_inches='tight')
    print("✅ 2D cluster plot saved")
    plt.close()

    # Cluster profile radar/bar chart
    features = ['age', 'annual_income', 'spending_score',
                'membership_years', 'purchase_frequency', 'last_purchase_amount']
    cluster_means = df.groupby('cluster')[features].mean()

    # Normalize for comparison
    cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    cluster_names = ['VIP Champions', 'Potential Loyalists', 'Budget Bargainers', 'At-Risk Shoppers']

    for i in range(4):
        ax = axes[i]
        bars = ax.bar(range(len(features)), cluster_means_norm.iloc[i],
                       color=colors[i], alpha=0.7, edgecolor='white')
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels([f.replace('_', '\n') for f in features], fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_title(f'Cluster {i}: {cluster_names[i]}', fontweight='bold')
        ax.set_ylabel('Normalized Score')

    plt.suptitle('Cluster Profiles Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../images/cluster_profiles.png', dpi=150, bbox_inches='tight')
    print("✅ Cluster profiles plot saved")
    plt.close()


def main():
    """Execute customer segmentation pipeline."""
    import os
    os.makedirs('../images', exist_ok=True)
    os.makedirs('../data', exist_ok=True)

    print("=" * 60)
    print("CUSTOMER SEGMENTATION - K-MEANS CLUSTERING")
    print("=" * 60)

    # Generate data
    print("\n🔄 Generating customer dataset...")
    df = generate_customer_data()
    df.to_csv('../data/customers.csv', index=False)
    print(f"✅ Dataset: {df.shape}")

    # Scale features
    features = ['age', 'annual_income', 'spending_score',
                'membership_years', 'purchase_frequency', 'last_purchase_amount']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    # Find optimal clusters
    best_k = find_optimal_clusters(X_scaled)

    # Apply clustering
    df, model = apply_clustering(df, X_scaled, n_clusters=best_k)

    # Visualize
    visualize_clusters(df, X_scaled)

    # Save results
    df.to_csv('../data/customers_segmented.csv', index=False)

    # Cluster naming
    cluster_names = {
        0: "VIP Champions",
        1: "Potential Loyalists",
        2: "Budget Bargainers",
        3: "At-Risk Shoppers"
    }

    print("\n" + "=" * 60)
    print("✅ SEGMENTATION COMPLETE!")
    print("=" * 60)
    print("\n📁 Generated files:")
    print("   - data/customers.csv")
    print("   - data/customers_segmented.csv")
    print("   - images/elbow_method.png")
    print("   - images/clusters_2d.png")
    print("   - images/cluster_profiles.png")
    print("\n🏷️ Cluster Labels:")
    for k, v in cluster_names.items():
        print(f"   Cluster {k}: {v}")


if __name__ == "__main__":
    main()
