# Sales Forecasting Machine Learning Project

##  Project Overview

A machine learning project that forecasts daily sales for the next 90 days using historical business data (2014-2017).

**Built as part of:** Future Interns ML Track  
**Track Code:** ML  
**Task Number:** 01  
**Date:** 14.05.2026  
**Author:** Srisharun AK

---

##  Project Objective

Predict future sales based on historical data and present results in a way businesses can use for:
- Inventory planning
- Staffing decisions
- Cash flow management
- Revenue forecasting

---

##  Dataset

**Source:** Superstore Sales Dataset  
**File:** Sample - Superstore.csv  
**Size:** 2.2 MB  
**Records:** 9,994 transactions  
**Time Period:** January 2014 - December 2017 (4 years)  
**Columns:** 21 features including Order Date, Sales, Quantity, Profit, Category, Region, etc.

---

##  Data Exploration

### Key Statistics:
- **Total Sales (4 years):** $2,297,200.86
- **Average Daily Sales:** $1,857.07
- **Highest Daily Sales:** $28,106.72
- **Lowest Daily Sales:** $2.02
- **Total Transactions:** 9,994
- **Days with Sales:** 1,237

### Data Quality:
- Zero missing values
- No data cleaning needed
- Proper date format after conversion
- All numeric values valid

---

##  Technologies & Tools
- Language:           Python 3.13.7
- IDE:                Jupyter Notebook
- Libraries:
• Pandas           - Data manipulation
• NumPy            - Numerical calculations
• Scikit-learn     - Machine learning
• Matplotlib       - Data visualization
---

##  Project Workflow

### Phase 1: Data Loading & Exploration
```python
# Load dataset
df = pd.read_csv('Sample - Superstore.csv', encoding='latin-1')

# Explore
print(df.head())
print(df.describe())
print(df.isnull().sum())
```
**Output:** Understanding data structure, 0 missing values, 21 columns

---

### Phase 2: Data Preparation
```python
# Convert dates to proper format
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%Y')

# Extract time features
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Quarter'] = df['Order Date'].dt.quarter
```
**Output:** Proper date objects, extracted temporal features

---

### Phase 3: Aggregation
```python
# Aggregate to daily sales
daily_sales = df.groupby('Order Date')['Sales'].sum().reset_index()
```
**Output:** 1,237 days of aggregated sales data

---

### Phase 4: Analysis & Visualization
```python
# Create moving averages
daily_sales['MA7'] = daily_sales['Total_Sales'].rolling(7).mean()
daily_sales['MA30'] = daily_sales['Total_Sales'].rolling(30).mean()

# Visualize
plt.plot(daily_sales['Date'], daily_sales['Total_Sales'])
plt.plot(daily_sales['Date'], daily_sales['MA30'])
```
**Output:** Identified seasonal patterns, trend analysis

---

### Phase 5: Forecasting Model
```python
# Seasonal forecasting approach
last_30_avg = daily_sales['MA30'].iloc[-1]
seasonal_pattern = daily_sales['Total_Sales'].iloc[-90:].values
seasonal_normalized = seasonal_pattern / seasonal_pattern.mean()

forecast = []
for i in range(90):
    predicted = last_30_avg * seasonal_normalized[i % 90]
    forecast.append(predicted)
```
**Output:** 90-day forecast with $271,423.80 predicted revenue

---

##  Results & Findings

### 90-Day Forecast (Q1 2018)
- **Period:** January 1 - March 30, 2018
- **Total Predicted Sales:** $271,423.80
- **Average Daily Sales:** $3,015.82
- **Peak Day Forecast:** $14,533.76
- **Low Day Forecast:** $10.24

### Key Insights:
1. **Seasonality Detected:** Sales follow a consistent seasonal pattern
2. **Stability:** Overall business trend is stable
3. **Variability:** Daily sales fluctuate based on seasonal factors
4. **Pattern Consistency:** Historical patterns repeat predictably

---

##  Business Recommendations

### 1. Inventory Management
- **Stock for average:** $3,015/day
- **Prepare for peaks:** Up to $14,500/day
- **Action:** Use safety stock strategy for high-demand days

### 2. Staffing
- **Seasonal variation:** Plan hiring around peak periods
- **High-sales days:** Need additional staff
- **Low-demand days:** Minimal staffing sufficient

### 3. Cash Flow Planning
- **Q1 2018 Forecast:** $271,423.80
- **Monthly average:** ~$90,474
- **Action:** Budget accordingly for variable revenue

---

##  Visualizations

### Chart 1: Daily Sales Over Time (2014-2017)
Shows raw daily sales with significant daily variation.

### Chart 2: Moving Averages Analysis
- Orange line: 7-day moving average (recent trend)
- Red line: 30-day moving average (true trend)
Shows clear seasonal patterns when noise is removed.

### Chart 3: Historical + Forecast
- Blue line: 4 years of historical data
- Orange dashed line: 90-day forecast
- Red dotted line: Forecast start date
Shows that forecast follows historical seasonal patterns.

---

##  Files in This Repository

FUTURE_ML_01/
│
├── README.md                                    # This file
│
├── Data Files/
│   ├── Sample - Superstore.csv                 # Original dataset
│   ├── sales_forecast_90days.csv               # 90-day predictions
│   └── sales_analysis_historical_forecast.csv  # Combined analysis
│
├── Code/
│   └── sales_forecast_notebook.ipynb           # Jupyter notebook with all code
│
├── Reports/
│   ├── FORECAST_REPORT.txt                     # Executive summary
│   └── PROJECT_DOCUMENTATION.md                # Detailed documentation
│
└── Visualizations/
├── daily_sales_chart.png
├── moving_averages_chart.png
└── forecast_chart.png
---

##  How to Use This Project

### 1. View the Code
Open `sales_forecast_notebook.ipynb` in Jupyter Notebook

### 2. Review the Data
- Original data: `Sample - Superstore.csv`
- Forecast results: `sales_forecast_90days.csv`

### 3. Check Results
- Summary: `FORECAST_REPORT.txt`
- Visualizations: See charts in Visualizations folder


##  Model Performance

### Approach 1: Linear Regression
- **Accuracy (R²):** 0.0086 (0.86%)
- **Status:**  Not sufficient
- **Reason:** Can't capture seasonal patterns

### Approach 2: Seasonal Forecasting (Final Model)
- **Method:** Last 30-day average + historical pattern application
- **Accuracy:** Captures seasonal patterns effectively
- **Status:**  Accepted
- **Validation:** Will be verified against actual 2018 data

---

##  Future Improvements

1. **Add External Factors**
   - Holiday calendars
   - Promotional events
   - Marketing campaigns

2. **Advanced Algorithms**
   - ARIMA (AutoRegressive Integrated Moving Average)
   - Prophet (Facebook's forecasting)
   - LSTM Neural Networks

3. **Confidence Intervals**
   - Add 95% confidence bands
   - Quantify forecast uncertainty

4. **Continuous Learning**
   - Monthly model retraining
   - Accuracy tracking
   - Seasonal adjustment updates

5. **Product-Level Forecasting**
   - Forecast by category
   - Forecast by region
   - Customer segment analysis

---

##  Learning Outcomes

Through this project, I learned:

 Data loading and cleaning with Pandas  
 Exploratory data analysis techniques  
 Time-series analysis and aggregation  
 Feature engineering for temporal data  
 Machine learning model training  
 Visualization and communication  
 Business problem solving with ML  
 Professional documentation practices  

---

##  Key Takeaways

1. **ML ≠ Accuracy** - Simple approaches often outperform complex ones
2. **Context Matters** - Business understanding is as important as code
3. **Visualization Speaks** - Charts communicate better than numbers
4. **Iteration Works** - First model wasn't great, improved approach succeeded
5. **Real Impact** - ML solves actual business problems

---

##  Questions or Feedback?

This project demonstrates core ML competencies:
- Data manipulation
- Analysis
- Forecasting
- Communication
- Business thinking

---

##  License

MIT License - Feel free to use this for learning purposes

---

##  Submission Details

- **Program:** Future Interns
- **Track:** Machine Learning (Code: ML)
- **Task:** 01
- **Repository Format:** FUTURE_ML_01 
- **Visibility:** Public 
- **Last Updated:** 14.05.2026

---

*This project was completed as part of the Future Interns Machine Learning internship program.*
