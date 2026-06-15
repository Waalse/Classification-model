# Drug Risk Clustering

This portfolio project demonstrates two approaches for classifying drugs into four risk categories:

- Low
- Medium
- High
- Critical

The project uses **synthetic drug supply-chain data** and contains no confidential records.

## Dataset

The dataset contains six columns:

| Column | Description |
|---|---|
| `drug` | Synthetic drug identifier |
| `alternatives` | Number of available alternatives |
| `shortage` | Number of historical shortage events |
| `inventory` | Current stock quantity |
| `demand` | Drug demand quantity |
| `supply` | Drug supply quantity |

## Methods

### Weighted risk method

The weighted method applies business-driven feature directions:

- Higher `shortage` and `demand` increase risk.
- Higher `alternatives`, `inventory`, and `supply` reduce risk.

A signed risk score is used to classify drugs into Low, Medium, High, and Critical groups.

### K-Means clustering

The same features are standardized and weighted by importance before applying K-Means with four clusters.

The clusters are automatically ordered from Low to Critical based on their average signed risk score.

## Notebook contents

The notebook includes:

- Data loading and validation
- Missing-value handling
- Feature scaling
- Directional weighting
- Weighted risk classification
- K-Means clustering
- Silhouette score
- Automatic cluster labeling
- Weighted-versus-ML agreement analysis
- Confusion matrix
- Cluster-center interpretation
- PCA visualization
- Results export

## Files

```text
drug-risk-clustering/
├── drug_data.csv
├── drug_risk_clustering.ipynb
└── README.md
```

## Run the project

Open `drug_risk_clustering.ipynb` in VS Code or Jupyter and run the cells in order.

Required Python libraries:

```bash
pip install pandas numpy scikit-learn matplotlib
```
