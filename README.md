# Insurance Risk Analytics

## Project Overview

This project develops a data-driven insurance risk analytics solution for AlphaCare Insurance. The objective is to analyze historical insurance policy and claims data, identify key risk factors that influence claim costs, and build predictive models that support risk-based pricing.

The project combines exploratory data analysis, statistical hypothesis testing, machine learning, and an interactive dashboard to help insurers better understand portfolio performance and make informed pricing decisions.

---

## Business Problem

Traditional insurance pricing often relies on broad customer segments, which may not accurately reflect individual risk. By leveraging historical policy and claims data, insurers can identify factors associated with higher claim costs and improve pricing strategies.

This project aims to answer questions such as:

* Which provinces present the highest insurance risk?
* Do customer demographics significantly affect claim severity?
* Can historical policy information be used to predict future claim costs?
* How can predictive analytics support fair and sustainable premium pricing?

---

## Project Objectives

The main objectives of this project are to:

* Explore and understand the insurance portfolio.
* Clean and preprocess insurance policy data.
* Perform exploratory data analysis (EDA).
* Conduct statistical hypothesis testing to validate business assumptions.
* Develop predictive models for claim severity.
* Compare multiple machine learning models.
* Build an interactive Streamlit dashboard for portfolio analysis and business insights.

---

## Dataset

The dataset contains historical insurance policy information, customer characteristics, vehicle details, premiums, and claims.

Examples of available features include:

* Province
* Postal Code
* Gender
* Vehicle Type
* Vehicle Model
* Vehicle Age
* Cover Type
* Sum Insured
* Premium Amount
* Claim Amount

Target Variable:

* **TotalClaims**

---

## Project Structure

```text
insurance_risk_analytics/
│
├── app/
│   └── app.py
│
├── data/
│   ├── MachineLearningRating_v3.txt
│   └── MachineLearningRating_cleaned.csv
│
├── models/
│   └── best_model.pkl
│
├── notebooks/
│   └── Insurance_Risk_Analytics.ipynb
│
├── reports/
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── hypothesis_testing.py
│   ├── modeling.py
│   └── visualization.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Week 12 Improvements

Compared with the original project, the following improvements were implemented:

- Refactored the project into a modular Python package.
- Improved statistical hypothesis testing with clearly defined hypotheses and summarized results.
- Added reusable preprocessing and modeling functions.
- Developed an interactive Streamlit dashboard for portfolio analysis.
- Improved project documentation with a comprehensive README.
- Enhanced project organization for maintainability and reproducibility.


## Project Workflow

### Task 1 — Data Preparation

* Loaded the raw insurance dataset.
* Cleaned missing and inconsistent values.
* Performed feature engineering.
* Prepared a machine learning ready dataset.

### Task 2 — Exploratory Data Analysis

Performed exploratory analysis to understand:

* Claim distributions
* Premium distributions
* Provincial risk
* Customer characteristics
* Vehicle characteristics
* Portfolio profitability

Business insights were generated to support pricing decisions.

### Task 3 — Statistical Hypothesis Testing

Business hypotheses were evaluated using statistical tests.

The analyses included:

* Province risk comparison
* Postal code risk comparison
* Postal code profit margin comparison
* Gender claim severity comparison

The results identified statistically significant and non-significant differences across customer groups.

### Task 4 — Predictive Modeling

Several regression models were developed and evaluated, including:

* Linear Regression
* Random Forest Regressor

Model performance was evaluated using:

* Root Mean Squared Error (RMSE)
* R² Score

The best-performing model was selected for deployment.

---

## Dashboard

An interactive Streamlit dashboard was developed to provide business users with portfolio insights.

Dashboard features include:

* Portfolio overview
* Total premium
* Total claims
* Portfolio profitability
* Loss ratio analysis
* Province risk visualization
* Claim severity distribution

### Dashboard Preview

> Add a screenshot after creating the `images` folder.

```markdown
![Dashboard](images/dashboard.png)
```

---

## Key Findings

Some important findings from the analysis include:

* Insurance risk varies across provinces.
* Geographic location has a measurable impact on claim frequency.
* Postal code differences were not statistically significant in the selected comparison.
* No significant difference in claim severity was observed between male and female policyholders.
* Portfolio loss ratio exceeded 100%, indicating that total claims were greater than total premium income during the observation period.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Streamlit
* Joblib
* Jupyter Notebook

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Dashboard

Start the Streamlit application:

```bash
streamlit run app/app.py
```

---

## Future Improvements

Potential enhancements include:

* Improving prediction performance using additional engineered features.
* Hyperparameter tuning for tree-based models.
* Deployment to a cloud platform.
* Real-time prediction using an API.
* Continuous model retraining using newly available claims data.

---

## Author

**Danayit**

Data & AI Practitioner

This project was completed as part of the KAIM (10 Academy) Data Science and AI Program.

