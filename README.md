# Customer Churn Prediction

An end-to-end machine learning project that predicts which telecom customers are likely to cancel their service, with an interactive Streamlit app for making predictions.

**Business problem:** Finding customers who are likely to churn lets a company reach out early with retention offers and reduce lost revenue.

---

## Table of Contents

1. [Results at a Glance](#results-at-a-glance)
2. [Dataset](#dataset)
3. [Tech Stack](#tech-stack)
4. [Project Structure](#project-structure)
5. [Getting Started](#getting-started)
6. [Methodology](#methodology)
7. [Key Insights](#key-insights)
8. [Streamlit App](#streamlit-app)
9. [Future Improvements](#future-improvements)
10. [Lessons Learned](#lessons-learned)
11. [Author](#author)
12. [License](#license)
13. [Acknowledgments](#acknowledgments)

---

## Results at a Glance

Best model: **Logistic Regression**, chosen for the highest ROC-AUC among five models.

| Metric | Score |
|---|---|
| ROC-AUC | **0.8458** |
| Accuracy | 80.41% |
| F1-Score | 0.5929 |

---

## Dataset

| | |
|---|---|
| **Source** | [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle, originally from IBM) |
| **Size** | 7,043 customers |
| **Features** | 20 |
| **Target** | `Churn` (Yes / No) |
| **Churn rate** | about 26.5% (imbalanced) |

**Feature groups**

| Group | Columns |
|---|---|
| Demographics | Gender, Senior Citizen, Partner, Dependents |
| Account | Tenure, Contract type, Payment method, Paperless billing |
| Services | Phone, Multiple lines, Internet, Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies |
| Charges | Monthly charges, Total charges |

---

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python 3.8+ |
| Machine learning | scikit-learn, XGBoost, imbalanced-learn |
| Data processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| App | Streamlit |
| Model storage | Joblib |

---

## Project Structure

```
churn-prediction-ml/
├── app.py                      # Streamlit app for live predictions
├── data/                       # Dataset (download from Kaggle)
├── models/
│   ├── best_model_*.pkl        # Trained best model
│   └── preprocessor.pkl        # Fitted encoders and scaler
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   └── 03model_training.ipynb  # Model training and comparison
├── src/
│   └── data_preprocessing.py   # Reusable preprocessing pipeline
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/AkshayBaldha01/<repo-name>.git
cd <repo-name>

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

4. Download the dataset from Kaggle and place it in the `data/` folder.

### Usage

| Task | Command |
|---|---|
| Explore the data | `jupyter notebook notebooks/01_eda.ipynb` |
| Train and compare models | `jupyter notebook notebooks/03model_training.ipynb` |
| Run the preprocessor on its own | `python src/data_preprocessing.py` |
| Launch the prediction app | `streamlit run app.py` |

---

## Methodology

### 1. Data preprocessing
- Fixed missing values in `TotalCharges`.
- Created three new features:

  | Feature | Meaning |
  |---|---|
  | `tenure_group` | Tenure grouped into bins |
  | `avg_monthly_per_tenure` | Average spend per month of tenure |
  | `num_services` | Number of subscribed services |

- Encoded categorical variables (binary and label encoding).
- Scaled numerical features with `StandardScaler`.

### 2. Model training
Five models were trained and compared:

1. Logistic Regression (baseline)
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. XGBoost

### 3. Evaluation
Models were compared on accuracy, precision, recall, F1-score and ROC-AUC. **ROC-AUC was the primary metric** because the classes are imbalanced.

### 4. Model selection
**Logistic Regression** had the highest ROC-AUC and a good balance of precision and recall. It is also simple and easy to explain to business users.

---

## Key Insights

- **Contract type** is the strongest predictor: month-to-month customers churn far more than customers on longer contracts.
- **Tenure** is inversely related to churn: customers in their first year churn the most.
- **Monthly charges:** higher bills go with higher churn.
- **Tech support:** customers with a tech support subscription churn less.

---

## Streamlit App

The app lets you enter a customer's details and returns:

- A **churn probability** shown on a risk gauge (low, medium or high)
- **Expected 12-month revenue at risk**
- The **risk factors** that stand out for that customer
- **Suggested retention actions**
- A chart of the **features that influenced the prediction most**

Two sample customers (high-risk and low-risk) can be loaded from the sidebar for a quick demo.

```bash
streamlit run app.py
```

---

## Future Improvements

- [x] Streamlit dashboard for predictions
- [ ] Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- [ ] Better class-imbalance handling (SMOTE / undersampling)
- [ ] REST API with FastAPI for model serving
- [ ] CI/CD pipeline for automated retraining
- [ ] Cloud deployment (AWS / Azure)

---

## Lessons Learned

- **Feature engineering** improved model performance.
- **Class imbalance** matters a lot in churn prediction and needs deliberate handling.
- **A simple model can win:** Logistic Regression beat the more complex tree-based models on ROC-AUC here.
- **Business context matters:** depending on the cost of losing a customer, optimizing for recall may be more valuable than accuracy.

---

## Author

**Akshay Baldha**
GitHub: [AkshayBaldha01](https://github.com/AkshayBaldha01) · LinkedIn: [akshay-baldha](https://www.linkedin.com/in/akshay-baldha-20a552188)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Dataset provided by IBM Watson Analytics.
- Inspired by real-world telecom churn challenges.