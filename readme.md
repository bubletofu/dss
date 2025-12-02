```mermaid

flowchart TD
    A["CLI Args<br/>--data, --budget, --out,<br/>--skip-kmeans, --opt-top-n,<br/>--model, --oversample"] --> B["Load CSV<br/>load_data()"]
    B --> C["ETL & Feature Engineering<br/>coerce_numeric, margin/log features,<br/>time & discount features"]
    C --> D["Train Classifier (LightGBM)<br/>train_logreg()"]
    D --> E["Scored Proposals<br/>add accept_prob = 1 - reject_prob"]

    E --> F{"Skip K-Means?<br/>(--skip-kmeans)"}
    F -- "No" --> G["Customer Segmentation<br/>train_kmeans()<br/>K-Means on behavior features"]
    F -- "Yes" --> H["Bypass segmentation<br/>keep scored data"]

    G --> I["Scored + Segmented Data<br/>df_seg with segment"]
    H --> I

    I --> J["Optimization<br/>optimize_knapsack()<br/>ILP (PuLP) with budget constraint"]
    J --> K["Save Outputs<br/>save_outputs()"]

    K --> L["CSV Outputs<br/>proposal_accept_scores.csv<br/>customer_segments.csv<br/>proposal_selection.csv"]
    K --> M["Model Artifact<br/>logreg_proposal.joblib"]
    K --> N["Text Report<br/>model_report.txt<br/>AUC / PR-AUC / k / profit"]

    L -.-> O["Power BI Dashboards<br/>load CSVs as data sources"]


```