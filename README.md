# 🛒 Customer Segmentation with K-Means Clustering

An unsupervised machine learning project that segments customers into distinct groups based on purchasing behavior and demographics. This analysis helps businesses tailor marketing strategies to each customer segment.

---

## 📌 Problem Statement

Understanding customer segments is essential for targeted marketing, personalized recommendations, and resource allocation. This project applies clustering algorithms to identify natural groupings in customer data.

## 🎯 Key Objectives

- Identify optimal number of customer segments using the Elbow Method and Silhouette Score
- Apply K-Means and Hierarchical Clustering
- Profile each segment with actionable characteristics
- Provide business recommendations for each cluster

## 🛠 Tech Stack

- **Python 3.10+**
- **Pandas** — Data manipulation
- **Scikit-Learn** — Clustering algorithms and preprocessing
- **SciPy** — Hierarchical clustering
- **Matplotlib & Seaborn** — Visualization

## 📊 Dataset

The dataset contains **5,000 customer records** with:

| Feature | Description |
|---------|-------------|
| `age` | Customer age |
| `annual_income` | Annual income in USD |
| `spending_score` | Spending score (1-100) |
| `membership_years` | Years as a member |
| `purchase_frequency` | Purchases per year |
| `last_purchase_amount` | Amount of last purchase |

## 📈 Results

**Optimal Clusters:** 4 (confirmed by Elbow Method and Silhouette Analysis)

| Cluster | Name | Characteristics | Strategy |
|---------|------|----------------|----------|
| 0 | 🌟 VIP Champions | High income, high spending, frequent purchases | Loyalty rewards, exclusive access |
| 1 | 💰 Potential Loyalists | High income, low spending | Targeted promotions, engagement campaigns |
| 2 | 🎯 Budget Bargainers | Low income, low spending | Discount offers, value bundles |
| 3 | ⚠️ At-Risk Window Shoppers | Low income, high spending score but low frequency | Retention campaigns, payment plans |

**Silhouette Score:** 0.52 (good separation)

## 🚀 Getting Started

```bash
pip install -r requirements.txt
python src/customer_segmentation.py
```

## 📁 Project Structure

```
customer-segmentation/
├── data/
│   └── customers.csv
├── src/
│   └── customer_segmentation.py
├── images/
│   ├── elbow_method.png
│   ├── clusters_2d.png
│   └── cluster_profiles.png
├── requirements.txt
└── README.md
```

## 🎓 What I Learned

- Determining optimal clusters using multiple validation methods
- Interpreting cluster characteristics for business value
- Dimensionality reduction with PCA for visualization
- Comparing hierarchical vs. partition-based clustering

---

**Status:** ✅ Completed | **Last Updated:** June 2026
