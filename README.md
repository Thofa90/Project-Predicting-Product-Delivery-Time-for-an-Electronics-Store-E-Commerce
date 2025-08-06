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

**🧹 Data Preprocessing Steps before Modeling**

The following preprocessing steps were applied to prepare the data for machine learning:

	1.	Dropped Unnecessary Columns: Removed columns like ID that don’t contribute to prediction.
	2.	Outlier Capping: Applied IQR method to cap extreme values in numerical features (e.g., prior purchases, weight, discount).
	3.	Skewness Handling: Applied log transformation to highly skewed columns like Discount_offered and Prior_purchases.
	4.	Categorical Encoding: Used Label Encoding to convert categorical variables (Warehouse_block, Mode_of_Shipment, Product_importance, Gender) into numeric form.
	5.	Train-Test Split: Split the dataset into training and testing sets using a standard 80-20 ratio for model evaluation.
	6.	Feature Scaling: Applied StandardScaler to normalize all numeric features and bring them to the same scale for consistent model performance.
