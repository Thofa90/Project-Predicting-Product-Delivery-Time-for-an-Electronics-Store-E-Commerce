# Project: Predicting Product Delivery Time for an Electronics Store E-Commerce
**📝 Project Summary**

This project focuses on predicting on-time vs. delayed deliveries using machine learning techniques in the e-commerce logistics domain. After performing thorough Exploratory Data Analysis (EDA), feature engineering, and model training, multiple classifiers including Logistic Regression, SVM, Decision Tree, Random Forest, K-Nearest Neighbors (KNN), and XGBoost were evaluated. The models were assessed based on Accuracy, Precision, Recall, F1-Score, and ROC-AUC, with a special emphasis on minimizing false negatives (missed late deliveries), which is critical for business impact.

After hyperparameter tuning, XGBoost emerged as the best-performing model, striking the optimal balance between Precision and Recall, and achieving the highest ROC-AUC score. Feature importance analysis revealed that Discount Offered, Prior Purchases, and Product Weight are the most influential variables affecting delivery delays. These insights can help businesses optimize logistics and improve customer satisfaction by proactively managing high-risk deliveries.

**1. Project Title**

Predicting Delivery Delays for E-Commerce Shipments

**2. Short Description**

This project builds a machine learning model to predict whether a delivery will be on time or delayed for an electronics store. It aims to help logistics teams reduce late deliveries based on data-driven insights.

**3. Table of Contents**

**🏢 Industry Overview & Business Context**

**📂 Data Loading & Goal of the Project**

**🔍 Exploratory Data Analysis**

	1.	Basic Information & Summary
	2.	Check for Missing & Duplicates
	3.	Target Variable Distribution
	4.	Numerical Features Analysis
	5.	Boxplots of Numerical Columns per Transaction
	6.	Boxplot vs Target Variable
	7.	Categorical Features Distribution
	8.	Correlation Heatmap
	9.	Multivariate Analysis
	10.	Correlation Heatmap with Numerical and Encoded Categorical Columns

**🧹 Data Preprocessing for Modeling**

	1.	Drop Unnecessary Column
	2.	Capping the Outliers
	3.	Handling Skewness of Numerical Columns
	4.	Encoding Categorical Variables
	5.	Train-test Split
	6.	Scale Numeric Features

**🤖 Modeling**

	1.	Baseline Modeling(Logistic Regression, SVM, Decision Tree, Random Forest, KNN, XGBoost)
	2.	Evaluation from Baseline Models
	3.	Hyperparameter Tuning All the Models
	4.	Evaluation after Hyperparameter Tuning
	5.	Feature Importance on Best Tuned Model

**✅ Conclusion**

**4. Motivation / Problem Statement**

Late deliveries lead to customer dissatisfaction and increased logistics cost. By predicting delays in advance, the company can proactively adjust shipping plans, prioritize resources, and potentially reduce the rate of late deliveries.

**5. Dataset Overview**

- ~11,000 transactions from an e-commerce electronics store
- Columns include:
  - `Warehouse_block`, `Mode_of_Shipment`, `Customer_care_calls`,
  - `Customer_rating`, `Cost_of_the_Product`, `Discount_offered`,
    `Weight_in_gms`, `Prior_purchase`,`product_Importance`,` Gender`,` Reached.on.Time_Y.N(target: 0 = on time, 1 = delayed)`

**6. Exploratory Data Analysis Summary (EDA)**

📊 Basic Information

	•	✅ Clean Dataset: No missing or duplicate values.
	•	⚠️ Target Imbalance: 60% delayed vs. 40% on-time deliveries.

🔢 Numerical Insights:

	•	Customer Care Calls: Mostly 3–4 calls; no outliers or link to delivery time.
	•	Customer Ratings: Evenly distributed from 1 to 5.
	•	Product Cost: Most items range from $150–$270; peaks around $260.
	•	Prior Purchases: Mostly between 2–5, showing moderate loyalty.
	•	Discount Offered: Highly skewed; most under 10%, outliers up to 65%.
	•	Weight in Grams: Bimodal; light (1000–2000g) and heavy (4000–5000g) products.

📦 Boxplot vs Target Variable:

	•	Higher Discounts → More Delays
	•	Heavier Products → More On-Time Deliveries
	•	Cost, Ratings, and Prior Purchases show little to no impact individually.

🏷️ Categorical Feature Insights:

	•	Warehouse Block F had the highest delay count.
	•	Shipping by Ship experienced more delays vs. Flight or Road.
	•	High Importance Products faced fewer delays.
	•	Gender showed no meaningful difference in delay frequency.
	•	High Discounts impacted delivery across all importance and customer types.

🔁 Correlation Summary:

	•	Strongest Delay Indicators:
	•	+0.40 → Higher discounts = more delays
	•	-0.27 → Heavier products = fewer delays
	•	Weak Influence: Customer behavior, product importance, and shipment type (individually).

📌 Final Notes:

	•	Multivariate patterns (e.g., light products + ship + high discount) are more predictive than individual features.
	•	EDA suggests that discount strategy and logistics optimization can significantly reduce delays.

**🧹 7. Data Preprocessing Steps before Modeling**

The following preprocessing steps were applied to prepare the data for machine learning:

	1.	Dropped Unnecessary Columns: Removed columns like ID that don’t contribute to prediction.
	2.	Outlier Capping: Applied IQR method to cap extreme values in numerical features (e.g., prior purchases, weight, discount).
	3.	Skewness Handling: Applied log transformation to highly skewed columns like Discount_offered and Prior_purchases.
	4.	Categorical Encoding: Used Label Encoding to convert categorical variables (Warehouse_block, Mode_of_Shipment, Product_importance, Gender) into numeric form.
	5.	Train-Test Split: Split the dataset into training and testing sets using a standard 80-20 ratio for model evaluation.
	6.	Feature Scaling: Applied StandardScaler to normalize all numeric features and bring them to the same scale for consistent model performance.

**8. Modeling & Evaluation**

I have trained multiple models:

- Logistic Regression
- Bagged models (Decision Tree, Random Forest)
- K‑Nearest Neighbors
- Support Vector Machine (SVM)
- XGBoost

Metrics evaluated:

- Accuracy
- Precision
- Recall (focus on catching delayed shipments)
- F1‑Score
- ROC‑AUC

Primary success criteria:

- High **recall** for delayed deliveries
- Balanced **F1‑score**

**9. Results & Interpretation**

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.63     | 0.67      | 0.74   | 0.70     | 0.72    |
| SVM                 | 0.66     | 0.83      | 0.54   | 0.65     | 0.73    |
| Decision Tree       | 0.68     | 0.97      | 0.48   | 0.64     | 0.74    |
| Random Forest       | 0.66     | 0.78      | 0.60   | 0.68     | 0.74    |
| KNN                 | 0.63     | 0.71      | 0.65   | 0.68     | 0.72    |
| **XGBoost (best)**  | 0.66     | 0.72      | **0.71**| **0.71**   | **0.75** |

XGBoost was the best model, balancing recall and F1‑score while achieving top ROC‑AUC.

**10. Feature Importance Insights**

📌 1. XGBoost Feature Importance

	•	Discount_offered is the most influential feature, contributing over 80% to the model’s decision-making.
	•	Prior_purchases and Weight_in_gms have a moderate influence.
	•	Features like Gender, Customer_rating, Mode_of_Shipment, and Warehouse_block have minimal impact.
	•	Importance is calculated based on gain, reflecting how much a feature improves split quality.

📌 2. Logistic Regression Coefficients

	•	Positive coefficients (e.g., Discount_offered) increase the likelihood of delay.
	•	Negative coefficients (e.g., Weight_in_gms, Prior_purchases) reduce the delay probability.
	•	Coefficient magnitude shows influence, helping to interpret linear feature impact.

**11. Usage Instructions**

Clone the repository:
```bash
git clone https://github.com/username/repo.git
cd repo
pip install -r requirements.txt
