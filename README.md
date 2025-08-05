# Project-Predicting-Product-Delivery-Time-for-an-Electronics-Store-E-Commerce
📝 Project Summary

This project focuses on predicting on-time vs. delayed deliveries using machine learning techniques in the e-commerce logistics domain. After performing thorough Exploratory Data Analysis (EDA), feature engineering, and model training, multiple classifiers including Logistic Regression, SVM, Decision Tree, Random Forest, K-Nearest Neighbors (KNN), and XGBoost were evaluated. The models were assessed based on Accuracy, Precision, Recall, F1-Score, and ROC-AUC, with a special emphasis on minimizing false negatives (missed late deliveries), which is critical for business impact.

After hyperparameter tuning, XGBoost emerged as the best-performing model, striking the optimal balance between Precision and Recall, and achieving the highest ROC-AUC score. Feature importance analysis revealed that Discount Offered, Prior Purchases, and Product Weight are the most influential variables affecting delivery delays. These insights can help businesses optimize logistics and improve customer satisfaction by proactively managing high-risk deliveries.
