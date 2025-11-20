# Africa Credit Challenge: Predicting Loan Default

This project tackles a real-world loan default prediction challenge using a dataset of loans from Kenya and Ghana. The goal was to build a machine learning model that can accurately identify high-risk loan applicants. The project covers the entire data science lifecycle: from in-depth exploratory data analysis and advanced feature engineering to model comparison and the creation of a practical, business-focused credit scoring system.

The final, champion model (XGBoost) was saved as a deployment-ready artifact.

---

## The Business Problem: Finding the Needle in the Haystack
![alt text](image.png)

The core challenge of this project lies in the **severe class imbalance** of the dataset. A staggering **98.2%** of loans were successfully paid back, while only a tiny fraction (**1.8%**) resulted in a default.

This creates two major problems:

1. **A "dumb" model can achieve 98% accuracy** by simply guessing "No Default" every time, making accuracy a useless metric. The true challenge is finding the rare defaulters.
    
2. The model's success must be measured by its ability to correctly identify this tiny minority of high-risk customers, making metrics like the **F1-Score** the primary indicator of performance.
    

## My Approach: A Step-by-Step guide

I followed a structured, end-to-end process to move from raw data to a functional, business-ready solution.

### 1. Exploratory Data Analysis (EDA)

I began with a deep dive into the data to uncover predictive patterns. The key findings were:

![alt text](image-1.png)
ustomers were found to be **over 10 times more likely to default** (a >20% default rate) compared to "Repeat Loan" customers (<2% default rate). This was the single strongest predictor in the dataset.

![alt text](image-2.png)

- **Loan Characteristics Matter:** Certain loan_type categories (like Type_15 and Type_23) had extremely high default rates, some exceeding 70%, identifying them as high-risk products.

![alt text](image-3.png)

- **Financial Behavior Signals Risk:** On average, customers who defaulted tended to take out larger loans and for longer durations than customers who paid back successfully.
    

### 2. Advanced Feature Engineering

The raw data was not enough. To provide the model with a richer context, I engineered several new, high-impact features:

- **Date-Based Features:** I extracted the month, day_of_week, and year from the loan disbursement date to capture potential seasonal trends or patterns.
    
- **Customer-Level Aggregation:** This was the most powerful feature engineering step. I grouped the entire dataset by customer_id to create a historical profile for each borrower. This generated features like:
    
    - customer_Total_Amount_mean (average loan size)
        
    - customer_duration_max (their longest loan duration)
        
    - customer_loan_type_count (total number of loans taken)
        

This process transformed the dataset from a simple list of loans into a rich collection of customer behaviors.

### 3. Data Preprocessing

To prepare the data for modeling, I performed two key steps:

- **Handled Missing Values:** My customer aggregation feature customer_Total_Amount_std correctly generated NaN values for customers with only one loan. I filled these with 0, as a standard deviation of zero is logical for a single data point.
    
- **One-Hot Encoded Categorical Features:** I converted categorical columns like loan_type and New_versus_Repeat into a numerical format that the models could understand.
    

### 4. Model Development and Evaluation

I built and rigorously evaluated a series of models, telling a clear story of iterative improvement:

1. **Baseline Model (Logistic Regression):** Established a starting F1-Score of **0.22**.
    
2. **Random Forest:** A more complex model that captured non-linear patterns, achieving a massive performance leap to an F1-Score of **0.57**.
    
3. **XGBoost (The Champion):** Using this industry-standard gradient boosting model, the performance jumped again to a phenomenal F1-Score of **0.76**.
    
4. **LightGBM & CatBoost:** I also tested these other powerful gradient boosting models, which performed well but ultimately confirmed that XGBoost was the top performer for this specific problem.
    

Throughout the process, I used **stratified k-fold cross-validation** to ensure my performance metrics were stable, reliable, and not a fluke.

### 5. From Prediction to Business Value: The Credit Scoring System

The most critical step of the project was translating the model's technical output (a probability like 0.7674) into a practical tool for loan officers. To do this, I designed a **5-tier credit scoring function** based on the distribution of the model's predicted default probabilities.

This system segments applicants into clear, actionable risk categories:

|   |   |   |   |
|---|---|---|---|
|Risk Category|Credit Score|Probability Range|Recommended Business Action|
|**Very Low Risk**|5 (Excellent)|0.0 - 0.003|Auto-Approve|
|**Low Risk**|4 (Good)|0.003 - 0.50|Standard Approval|
|**Medium Risk**|3 (Fair)|0.50 - 0.90|Requires Manual Underwriter Review|
|**High Risk**|2 (Poor)|0.90 - 0.98|Likely to be Rejected|
|**Very High Risk**|1 (Very Poor)|> 0.98|Auto-Reject|

This transforms the machine learning model from a simple classifier into an automated decision-support tool that can drive business efficiency and reduce risk.