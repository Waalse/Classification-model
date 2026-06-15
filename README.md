# Drug Risk Clustering

A machine learning project that classifies drugs into four supply-chain risk categories — **Low, Medium, High, and Critical** — using synthetic drug supply-chain data.

The project combines domain-driven feature engineering, automated weight optimisation, and multiple clustering algorithms to produce interpretable, actionable risk tiers.

---

## Dataset

The dataset (`drug_data.csv`) contains **1,200 synthetic drug records** with no confidential information.

| Column | Type | Description |
|---|---|---|
| `drug` | string | Synthetic drug identifier |
| `alternatives` | integer | Number of available substitute drugs |
| `shortage` | integer | Number of historical shortage events |
| `inventory` | integer | Current stock quantity |
| `demand` | integer | Drug demand quantity |
| `supply` | integer | Drug supply quantity |

---

## Feature Engineering

Three derived features are computed to capture non-linear supply-chain risk signals:

| Feature | Formula | Risk direction |
|---|---|---|
| `supply_coverage` | `inventory / (demand + 1)` | Higher = lower risk (stock lasts longer) |
| `stockout_risk` | `max(0, demand - supply) / (demand + 1)` | Higher = higher risk (unmet demand fraction) |
| `shortage_severity` | `shortage / (alternatives + 1)` | Higher = higher risk (shortage with few substitutes) |

---

## Methods

### Directional weighting

Each feature is assigned a signed weight reflecting its risk direction:

- **Positive weights** — `shortage`, `demand`, `stockout_risk`, `shortage_severity` (higher values increase risk)
- **Negative weights** — `alternatives`, `inventory`, `supply`, `supply_coverage` (higher values reduce risk)

The signed weighted sum of standardised features produces a continuous `Risk_Direction_Score` per drug, which is used to rank clusters from Low to Critical.

### Weight optimisation

A random search over 3,000 weight combinations is run automatically. Signs are fixed by feature direction — only magnitudes are sampled. Each candidate set is scored by Silhouette Score. A balance constraint (no single cluster > 50% or < 5% of drugs) prevents the score from being gamed by one dominant cluster.

### Clustering algorithms

Both algorithms are run on the same weighted feature space and compared across three metrics:

| Algorithm | Description |
|---|---|
| **K-Means** | Centroid-based, minimises within-cluster variance |
| **Gaussian Mixture Model (GMM)** | Probabilistic, allows soft cluster assignments and flags borderline drugs |

The algorithm with the higher Silhouette Score is automatically selected for final labelling.

### Scaling

`RobustScaler` is used instead of `StandardScaler` because `inventory` and `demand` have heavy-tailed distributions. RobustScaler uses the interquartile range and is not distorted by outliers.

---

## Clustering Quality Metrics

| Metric | Better when | What it measures |
|---|---|---|
| Silhouette Score | Higher (max 1) | How well each drug fits its own cluster vs the nearest other cluster |
| Davies-Bouldin Score | Lower (min 0) | Ratio of within-cluster scatter to between-cluster separation |
| Calinski-Harabasz Score | Higher | Ratio of between-cluster dispersion to within-cluster dispersion |

---

## Notebooks

### `drug_risk_clustering_directional_weights.ipynb`

The main enhanced notebook. Runs end-to-end:

1. Data loading and cleaning
2. Derived feature engineering
3. RobustScaler normalisation
4. Directional weight assignment
5. Baseline K-Means clustering and metrics
6. Automated weight random search
7. Optimised K-Means re-run
8. GMM with soft assignments and borderline drug flagging
9. Algorithm comparison table (all three metrics, best highlighted)
10. Risk level assignment via cluster ranking
11. Cluster distribution summary
12. Cluster centers in original scale (all 8 features)
13. PCA scatter plot
14. Radar chart — cluster risk profiles
15. Results export

### `drug_risk_clustering(1).ipynb`

A comparison notebook that benchmarks the ML clustering approach against a transparent percentile-based weighted risk method. Includes a confusion matrix showing agreement between the two methods, and compares K-Means against Agglomerative Clustering (Ward linkage).

---

## Outputs

| File | Description |
|---|---|
| `clustered_drug_results.csv` | Full dataset with cluster assignments, risk scores, and risk levels |
| `pca_clusters_optimised.png` | PCA scatter plot coloured by risk level |
| `cluster_radar_profiles.png` | Radar chart showing feature profiles per risk tier |
| `outputs/clustered_drug_results.csv` | Results from the comparison notebook |
| `outputs/method_confusion_matrix.csv` | Agreement matrix between ML and weighted methods |
| `outputs/weighted_cluster_centers.csv` | Cluster centers in standardised space |
| `outputs/original_scale_cluster_centers.csv` | Cluster centers in original feature units |

---

## File Structure

```
drug-risk-clustering/
├── drug_data.csv
├── drug_risk_clustering_directional_weights.ipynb   ← main notebook
├── drug_risk_clustering(1).ipynb                    ← comparison notebook
├── clustered_drug_results.csv
├── pca_clusters_optimised.png
├── cluster_radar_profiles.png
├── outputs/
│   ├── clustered_drug_results.csv
│   ├── method_confusion_matrix.csv
│   ├── weighted_cluster_centers.csv
│   └── original_scale_cluster_centers.csv
└── README.md
```

---

## Requirements

```bash
pip install pandas numpy scikit-learn matplotlib
```

| Library | Version tested |
|---|---|
| pandas | >= 1.5 |
| numpy | >= 1.23 |
| scikit-learn | >= 1.2 |
| matplotlib | >= 3.6 |

---

## How to Run

Open `drug_risk_clustering_directional_weights.ipynb` in VS Code or Jupyter and run all cells in order. The weight optimisation cell (Cell 9) takes approximately 30 seconds.

To adjust the risk distribution or weight search ranges, edit the parameters at the top of cells 9 and 13.
