# Heart Disease Prediction – Random Forest

Predict heart disease from clinical patient data using a **Random Forest Classifier**.  
This project emphasizes **clinically aware preprocessing**, **robust modeling**, and **deployment-ready logic** for new patient prediction.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Preprocessing](#preprocessing)  
- [Modeling](#modeling)  
- [Evaluation](#evaluation)  
- [Feature Importance](#feature-importance)  
- [New Patient Prediction](#new-patient-prediction)  
- [Clinical Interpretation](#clinical-interpretation)  
- [Next Steps](#next-steps)  
- [Requirements](#requirements)  
- [License](#license)  

---

## Project Overview

This project predicts the presence of **heart disease** using the Cleveland Heart Disease dataset.  

Key features:

- **Random Forest Classifier**  
- Median imputation for clinically invalid zero values  
- StandardScaler for feature scaling  
- Preprocessing and model applied consistently to new patient data  

The project is **research-focused and deployment-ready**, intended to assist doctors rather than replace clinical judgment.

---

## Dataset

- Source: `heart.csv` (Cleveland dataset)  
- Target column: `target`  
  - `0` = Healthy  
  - `1` = Heart Disease  

Features include:

| Feature | Description |
|---------|-------------|
| age     | Age in years |
| sex     | 1 = male, 0 = female |
| cp      | Chest pain type (0–3) |
| trestbps| Resting blood pressure (mm Hg) |
| chol    | Serum cholesterol (mg/dl) |
| fbs     | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| restecg | Resting electrocardiographic results (0–2) |
| thalach | Maximum heart rate achieved |
| exang   | Exercise-induced angina (1 = yes, 0 = no) |
| oldpeak | ST depression induced by exercise |
| slope   | Slope of ST segment (0–2) |
| ca      | Number of major vessels colored by fluoroscopy (0–3) |
| thal    | Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect) |

---

## Preprocessing

1. **Train-Test Split**  
   - 80% training / 20% testing  
   - Stratified by target to preserve class balance  

2. **Median Imputation**  
   - Columns that cannot be zero clinically: `trestbps`, `chol`, `thalach`, `oldpeak`  
   - Median calculated **only from training set**  

3. **Feature Scaling**  
   - StandardScaler applied after imputation  
   - Ensures all features contribute equally  

> Same preprocessing applied to **new patient input**.

---

## Modeling

python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
rf.fit(X_train_scaled, y_train)

## Evaluation

The model achieves perfect separation on the test set:
Confusion Matrix:
[[100   0]
 [  0 105]]

Recall (Sensitivity): 1.0
Interpretation: All healthy and heart disease patients were correctly classified on the test set.
⚠️ Real-world performance may vary; external validation is required.
## Feature Importance

Visualize most predictive clinical features:
import matplotlib.pyplot as plt

plt.barh(X.columns, rf.feature_importances_)
plt.xlabel("Importance")
plt.title("Feature Importance (Clinical Relevance)")
plt.show()
Allows doctors and researchers to see which clinical variables most influence predictions.
## New Patient Prediction

Example for a new patient:
new_patient = pd.DataFrame([{
    'age': 55, 'sex': 1, 'cp': 2, 'trestbps': 130, 'chol': 250,
    'fbs': 0, 'restecg': 1, 'thalach': 150, 'exang': 0,
    'oldpeak': 1.2, 'slope': 2, 'ca': 0, 'thal': 2
}])

# Apply same preprocessing
new_patient[columns_with_invalid_zeros] = imputer.transform(new_patient[columns_with_invalid_zeros])
new_patient_scaled = scaler.transform(new_patient)

prediction = rf.predict(new_patient_scaled)
probability = rf.predict_proba(new_patient_scaled)

print("Prediction (0=Healthy, 1=Heart Disease):", prediction[0])
print("Probability of heart disease:", probability[0][1])
## Clinical Interpretation

prediction = 1 → Model suspects heart disease

probability → Confidence level (e.g., 0.55 = moderate)

Decision is always finalized by a doctor, model assists diagnosis
## Next Steps

ROC-AUC curve & calibration

SHAP explainability for feature-level insight

External validation on other heart disease datasets

Web app deployment for clinical use
## Requirements
pandas
numpy
scikit-learn
matplotlib

## License

MIT License – free for research and educational use.
