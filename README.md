# Bike Demand Intelligence

## Project Overview

Bike Demand Intelligence is a simple Python-based Data Science and AI/ML decision-support project.

The project follows the Business Intelligence flow discussed in the IBM SkillsBuild Data Analytics with AI internship:

**KPIs → Trends → Drivers → Risk/Opportunity → Action**

The goal is to understand daily bike-rental demand and convert the analysis into practical planning insights.

## Problem Statement

Bike-sharing operators need to understand when demand is high or low and which environmental or calendar-related factors are useful for predicting demand. This project analyzes historical bike-rental data and provides a simple demand prediction model and business recommendations.

## Objectives

1. Understand overall bike-rental performance.
2. Identify useful KPIs.
3. Analyze daily, monthly and seasonal demand trends.
4. Identify important predictive drivers.
5. Predict daily rental demand using machine learning.
6. Identify planning risks and opportunities.
7. Convert analytical findings into practical actions.

## Dataset

**Dataset:** Bike Sharing Dataset

**Source:** UCI Machine Learning Repository

**Dataset link:** https://archive.ics.uci.edu/dataset/275/bike%2Bsharing%2Bdataset

The application downloads the daily dataset automatically when it starts.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit
- Random Forest Regression

## Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
KPI Analysis
   ↓
Trend Analysis
   ↓
Driver Analysis
   ↓
Risk & Opportunity
   ↓
Random Forest Prediction
   ↓
Business Actions
```

## AI / ML Component

A Random Forest Regression model is used to predict daily bike-rental demand.

The model uses calendar, weather and temperature-related variables. The data is sorted chronologically and divided into an 80% training period and a 20% test period.

The project reports:

- MAE
- RMSE
- R²
- Feature importance
- Actual vs predicted demand

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the application

```bash
streamlit run Afroz_BikeDemandIntelligence.py
```

### 3. Open the local URL

Streamlit will display a local URL, normally:

```text
http://localhost:8501
```

## Dashboard Sections

### Executive Overview
Shows the main KPIs and overall demand trend.

### Trend Analysis
Shows monthly and seasonal demand patterns.

### Driver Analysis
Shows the most important variables used by the prediction model.

### Risk & Opportunity
Translates demand variation into planning insights.

### AI / ML Demand Prediction
Shows model performance and actual vs predicted demand.

### Action
Provides practical actions based on the analysis.

## Important Note

The model's feature importance indicates predictive contribution in the trained model. It should not be interpreted as proof that a variable directly causes demand to change.

## Project Submission Files

The internship submission requires:

1. Code file
2. requirements.txt
3. Project Report
4. README.md

The project repository should contain these files along with the GitHub repository link used for submission.
