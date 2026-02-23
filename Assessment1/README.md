# ASSESSMENT 1
# Customer Churn Prediction

## Project Overview
This project focuses on analyzing customer churn — the phenomenon where customers stop using a company’s products or services. By leveraging machine learning techniques, the goal is to identify patterns in customer behavior and predict churn, enabling businesses to take proactive measures to retain customers.

## Repository Structure
```
├── data/                 # Contains the raw and processed data files
├── notebook/             # Jupyter Notebook(s) with data exploration, modeling, and results
└── README.md             # Project documentation
```

- **data/**: Includes the customer churn dataset used for training and evaluation.  
- **notebook/**: Contains the Jupyter Notebook with step-by-step analysis, feature engineering, model training, and evaluation.  

## Requirements
To run the notebook, install the following dependencies:

- Python 3.8+
- Jupyter Notebook / JupyterLab
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

You can install them via:
```bash
pip install -r requirements.txt
```

*(If you don’t have a `requirements.txt` yet, you can generate one with `pip freeze > requirements.txt`.)*

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Jans-AIML/AML-3303.git
   cd customer-churn
   ```
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook notebook/ConnectWave_Customer_Churn_Prediction_c0936855.ipynb
   ```
3. Run the cells sequentially to reproduce the analysis and results.

## Key Steps in the Notebook
- **Data Exploration**: Summary statistics, missing values, and churn distribution.  
- **Feature Engineering**: Encoding categorical variables, scaling numerical features.  
- **Modeling**: Training machine learning models (e.g., Random Forest, XGBoost).  
- **Evaluation**: Comparing models using accuracy, precision, recall, F1-score, and ROC-AUC.  
- **Insights**: Highlighting the most important features influencing churn.  

## Outcomes
- A reproducible workflow for churn prediction.  
- Benchmark results across multiple models.  
- Actionable insights for customer retention strategies.  

## Next Steps
- Extend the dataset with additional customer attributes.  
- Experiment with deep learning models.  
- Deploy the best-performing model as an API or dashboard.  

---
