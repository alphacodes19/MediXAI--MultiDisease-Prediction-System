# MediXAI - Unified Multi-Disease Prediction System

A Streamlit web application that unifies three clinical machine learning models for disease
risk prediction with Explainable AI (SHAP + LIME), an AI chatbot, bulk CSV processing,
OCR report analysis, personalised recommendations, and PDF export.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [File Structure](#2-file-structure)
3. [How to Run](#3-how-to-run)
4. [The Three Disease Models](#4-the-three-disease-models)
   - [Diabetes (NHANES)](#41-diabetes-nhanes)
   - [Heart Disease (UCI)](#42-heart-disease-uci)
   - [Parkinson's (Oxford Voice)](#43-parkinsons-oxford-voice)
5. [Explainable AI — SHAP and LIME](#5-explainable-ai--shap-and-lime)
6. [Feature Explanations](#6-feature-explanations)
7. [Observation Images Explained](#7-observation-images-explained)
8. [Application Features](#8-application-features)
9. [Database Design](#9-database-design)
10. [How to Add or Modify Things](#10-how-to-add-or-modify-things)
11. [Known Limitations](#11-known-limitations)
12. [Disclaimer](#12-disclaimer)

---

## 1. Project Overview

MediXAI is built on top of three independently developed ML projects:

| Source Project | Disease | Best Model | Accuracy |
|---|---|---|---|
| diabetes_pred_xai | Diabetes | RandomForestClassifier | ~96% |
| part_two / heart notebook | Heart Disease | ExtraTreesClassifier | ~91.3% |
| part_two / Parkinson's notebook | Parkinson's Disease | XGBoost (tuned) | ~95%+ |

The unified portal loads all three models and adds:
- Explainability (SHAP waterfall + importance + LIME)
- Login/Register with per-user data isolation
- History tracking with trend charts
- SHAP-driven + LLM-generated health recommendations
- AI chatbot with full prediction context injection
- OCR report upload with automatic AI analysis
- Bulk CSV batch prediction
- PDF report generation

---

## 2. File Structure

```
MediXAI/
|
|-- app.py                          ENTRY POINT — run this
|-- requirements.txt                All Python packages needed
|-- medixai_history.db              SQLite database (auto-created on first run)
|-- .env                            Optional: store GROQ_API_KEY here
|
|-- models/                         Trained model files (DO NOT EDIT)
|   |-- diabetes/
|   |   |-- diabetes_model.pkl      RandomForestClassifier (trained on NHANES)
|   |   |-- scaler.pkl              StandardScaler for diabetes features
|   |   |-- nhanes_diabetes_clean.xls  Training data (used as LIME background)
|   |
|   |-- heart/
|   |   |-- saved_models/
|   |       |-- extra_trees.pkl     ExtraTreesClassifier (best of 17 models)
|   |       |-- scaler.pkl          StandardScaler (used only for SVM model)
|   |       |-- heart_lime_bg.npy   Optional: X_train for LIME (generate from notebook)
|   |
|   |-- parkinsons/
|       |-- saved_models/
|           |-- best_model.pkl      Winning model from notebook comparison
|           |-- scaler.pkl          RobustScaler
|           |-- feature_names.json  MI-selected feature list
|           |-- xgboost_tuned.pkl   XGBoost tuned model
|           |-- random_forest_tuned.pkl
|           |-- svm_tuned.pkl
|
|-- pages/                          One file per app page
|   |-- __init__.py
|   |-- diabetes.py                 Diabetes prediction form + SHAP + LIME
|   |-- heart.py                    Heart disease form + SHAP + LIME
|   |-- parkinsons.py               Parkinson's voice feature form + SHAP + LIME
|   |-- bulk_csv.py                 Batch CSV upload and prediction
|   |-- ocr.py                      OCR report upload + AI analysis + autofill
|   |-- history.py                  Prediction history charts and table
|   |-- recommendations.py          SHAP-driven rules + Groq LLM plan
|   |-- chatbot.py                  Multi-session AI chatbot
|
|-- utils/                          Shared utility modules
|   |-- __init__.py
|   |-- database.py                 All SQLite operations (users, predictions, chats)
|   |-- models.py                   Model loading (cached) + shared helpers
|   |-- xai.py                      SHAP and LIME chart rendering
|   |-- llm.py                      Groq API calls + context building
|   |-- pdf_export.py               PDF report generation with fpdf2
|
|-- notebooks/                      Original training notebooks (read-only reference)
|   |-- 01_diabetes/
|   |   |-- Data_Download.ipynb     Downloads NHANES data using CDC API
|   |   |-- Dataset_Create.ipynb    Cleans and merges NHANES tables
|   |   |-- Train_model.ipynb       Trains RandomForest, saves pkl files
|   |
|   |-- 02_heart/
|   |   |-- heart_disease_comprehensive_XAI.ipynb   Trains 17 models, saves best
|   |
|   |-- 03_parkinsons/
|       |-- parkinson_prediction_enhanced_XAI.ipynb  Full pipeline with XAI
```

---

## 3. How to Run

### Step 1 - Install Python (3.10 or higher required)
```bash
python --version
```

### Step 2 - Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### Step 3 - Install packages
```bash
pip install -r requirements.txt
```

### Step 4 - Install Tesseract (for OCR feature only)
Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
Mac:     brew install tesseract
Linux:   sudo apt install tesseract-ocr

If you skip this step, everything works EXCEPT the OCR Report Upload page.

### Step 5 - Set your Groq API key
Open app.py and find line 28:
```python
GROQ_API_KEY = ""
```
Paste your key between the quotes. Get a free key at https://console.groq.com

### Step 6 - Run
```bash
streamlit run app.py
```
Opens at http://localhost:8501

### Step 7 - Register an account
When you first open the app, you will see a Login/Register screen.
Click Register, choose a username and password, then login.

---

## 4. The Three Disease Models

### 4.1 Diabetes (NHANES)

**Dataset:** National Health and Nutrition Examination Survey (NHANES)
Source: US Centers for Disease Control and Prevention (CDC)
Downloaded using the NHANES API in Data_Download.ipynb.

**Features (7):**

| Feature Code | Name | Description |
|---|---|---|
| RIDAGEYR | Age | Patient age in years |
| RIAGENDR | Gender | 1 = Male, 2 = Female |
| BMXBMI | BMI | Body Mass Index |
| DBP_mean | Diastolic BP | Average diastolic blood pressure (mmHg) |
| SBP_mean | Systolic BP | Average systolic blood pressure (mmHg) |
| LBXGLU | Glucose | Fasting blood glucose (mg/dL) |
| LBXGH | HbA1c | Glycated haemoglobin percentage |

**Target:** diabetes_label (0 = no diabetes, 1 = diabetes)

**Model:** RandomForestClassifier
- Grid search over n_estimators, max_depth, min_samples_split, min_samples_leaf
- class_weight='balanced' to handle class imbalance
- StandardScaler applied before training

**Clinical thresholds:**
- Glucose < 100 mg/dL = Normal
- Glucose 100-125 = Pre-diabetic
- Glucose >= 126 = Diabetic
- HbA1c < 5.7% = Normal
- HbA1c 5.7-6.4% = Pre-diabetic
- HbA1c >= 6.5% = Diabetic


### 4.2 Heart Disease (UCI)

**Dataset:** UCI Heart Disease dataset (918 patients)
Source: Combination of Cleveland, Hungarian, Switzerland, and VA datasets.

**Features (14 = 11 raw + 3 engineered):**

Raw features:
| Feature | Description |
|---|---|
| Age | Patient age |
| Sex | 0 = Male, 1 = Female (after encoding) |
| ChestPainType | 0=Typical Angina, 1=Atypical, 2=Non-Anginal, 3=Asymptomatic |
| RestingBP | Resting blood pressure (mmHg) |
| Cholesterol | Serum cholesterol (mg/dL) — 0 values KNN-imputed |
| FastingBS | Fasting blood sugar > 120 mg/dL (0/1) |
| RestingECG | 0=Normal, 1=ST-T abnormality, 2=LV hypertrophy |
| MaxHR | Maximum heart rate achieved |
| ExerciseAngina | Exercise-induced angina (0=No, 1=Yes) |
| Oldpeak | ST depression induced by exercise |
| ST_Slope | Slope of peak exercise ST segment (0=Up, 1=Flat, 2=Down) |

Engineered features (computed automatically at prediction time):
| Feature | Formula | Why |
|---|---|---|
| Age_MaxHR_ratio | Age / (MaxHR + 1) | Captures cardiovascular reserve |
| BP_Chol_ratio | RestingBP / (Cholesterol + 1) | Combined metabolic stress |
| OldpeakAbs | abs(Oldpeak) | Makes negative values comparable |

**Preprocessing:** LabelEncoder for categorical columns, KNNImputer for zero Cholesterol/RestingBP.

**Model selection:** 17 models were compared including:
- Individual: Logistic Regression, SVM (RBF), SVM (Linear), KNN, Naive Bayes
- Tree: Decision Tree, Random Forest, Extra Trees, Bagging
- Boosting: AdaBoost, Gradient Boosting, HistGradient Boosting
- Ensemble: Voting (hard), Voting (soft), Stacking (LR meta), Stacking (GBM meta)

Winner: ExtraTreesClassifier (n_estimators=500, max_depth=8, min_samples_leaf=2)
Accuracy: ~91.3% | 10-fold CV confirms stability

**Why ExtraTrees beats others:**
ExtraTrees uses extra randomisation — it selects both the feature AND the split point
randomly, then picks the best. This reduces variance more than Random Forest while
maintaining similar bias, making it robust on this dataset.


### 4.3 Parkinson's (Oxford Voice)

**Dataset:** Oxford Parkinson's Disease Detection dataset
195 voice recordings, 147 Parkinson's / 48 healthy.
23 voice biomarkers extracted from sustained phonations of "aaah".

**Feature engineering (adds 3 features before selection):**
```
Freq_Range       = MDVP:Fhi(Hz) - MDVP:Flo(Hz)   (frequency spread)
Freq_Variability = MDVP:Fhi(Hz) / MDVP:Fo(Hz)    (relative freq change)
NHR_HNR_ratio    = NHR / (HNR + 1e-6)             (noise-to-harmonic ratio)
```

**Feature selection via Mutual Information (MI):**
MI measures how much knowing a feature reduces uncertainty about the label.
Top 75% of features (MI > 25th percentile) are kept.
The selected list is saved to feature_names.json at training time.

**Scaler:** RobustScaler (uses median and IQR instead of mean and std)
Better than StandardScaler for voice data because voice biomarkers have
significant outliers from speech irregularities.

**Model comparison:** XGBoost, Random Forest, SVM all tuned with RandomizedSearchCV.
Stacking and Voting ensembles also tested.
Best model is saved as best_model.pkl.

**Voice biomarker groups:**

Jitter (frequency variation, cycle-to-cycle):
- MDVP:Jitter(%) — percentage jitter
- MDVP:Jitter(Abs) — absolute jitter in microseconds
- MDVP:RAP — relative average perturbation
- MDVP:PPQ — five-point period perturbation quotient
- Jitter:DDP — average absolute difference of differences between consecutive periods

Shimmer (amplitude variation):
- MDVP:Shimmer — local shimmer
- MDVP:Shimmer(dB) — local shimmer in dB
- Shimmer:APQ3, APQ5, APQ11 — amplitude perturbation quotients
- Shimmer:DDA — average absolute differences between consecutive differences

Nonlinear dynamics (voice complexity):
- RPDE — recurrence period density entropy
- DFA — detrended fluctuation analysis (scaling exponent)
- spread1, spread2 — nonlinear measures of fundamental frequency variation
- D2 — correlation dimension
- PPE — pitch period entropy

---

## 5. Explainable AI - SHAP and LIME

### Why XAI?
A prediction alone ("you have 68% diabetes risk") doesn't help you understand WHY.
XAI shows which specific features drove that number up or down.

### SHAP (SHapley Additive exPlanations)

SHAP is based on game theory (Shapley values). It answers:
"How much did each feature contribute to this prediction?"

For tree-based models (RF, ExtraTrees, XGBoost), TreeExplainer is used.
It is exact (not an approximation) and very fast.

Reading a SHAP waterfall chart:
- Each bar represents one feature
- RED bar (positive value) = this feature pushed the prediction TOWARDS disease
- GREEN bar (negative value) = this feature pushed the prediction AWAY from disease
- Bar length = how strongly that feature influenced THIS specific prediction
- The values are additive: base_value + sum(all SHAP values) = final prediction probability

Example for diabetes:
- Glucose = 135 mg/dL → SHAP +0.15 (increases risk significantly)
- Age = 35 years → SHAP -0.04 (young age decreases risk slightly)
- HbA1c = 7.2% → SHAP +0.22 (very high — biggest driver)

### LIME (Local Interpretable Model-agnostic Explanations)

LIME works differently from SHAP. For one patient:
1. It creates hundreds of perturbed versions of the patient's values
2. Runs all of them through the model
3. Fits a simple linear model to explain the model's behaviour locally

Reading a LIME bar chart:
- Each bar is a condition string like "Glucose > 110"
- ORANGE bar = this condition pushes prediction TOWARDS disease
- BLUE bar = this condition pushes prediction AWAY from disease

LIME background data:
- Diabetes: uses real NHANES data (2000 patients) for realistic perturbations
- Heart: needs heart_lime_bg.npy (X_train from notebook) — zeros used as fallback
- Parkinson's: needs parkinsons_lime_bg.npy — zeros used as fallback

To generate heart_lime_bg.npy, add this to your heart notebook:
```python
import numpy as np
np.save("heart_lime_bg.npy", X_train.values)
```
Then copy heart_lime_bg.npy to models/heart/saved_models/

### SHAP vs LIME comparison
| | SHAP | LIME |
|---|---|---|
| Method | Game theory (Shapley) | Local linear approximation |
| Speed | Fast (TreeExplainer) | Slower (needs many samples) |
| Accuracy | Exact for tree models | Approximate |
| Output | Numeric SHAP values | Condition strings + weights |
| Best for | "How much did each feature contribute?" | "What value ranges drove this?" |

---

## 6. Feature Explanations

### Diabetes features in plain English

**Glucose (LBXGLU):** Blood sugar level after fasting for 8+ hours.
The most direct indicator of diabetes. The pancreas should produce enough insulin
to keep glucose below 100 mg/dL. When it can't, glucose builds up.

**HbA1c (LBXGH):** A "3-month average" of blood glucose. Red blood cells live ~90 days.
Glucose sticks to haemoglobin. HbA1c measures what percentage of haemoglobin is glycated.
More reliable than a single glucose test because it reflects long-term control.

**BMI (BMXBMI):** Body mass index = weight(kg) / height(m)^2.
Obesity (BMI >= 30) causes insulin resistance — cells don't respond to insulin properly.

**Blood pressure (SBP/DBP):** Hypertension and diabetes often co-occur and worsen each other.
High BP damages blood vessels, compounding the vascular damage from high glucose.

### Heart disease features in plain English

**Cholesterol:** Low-density lipoprotein (LDL) builds up in artery walls, forming plaques.
Over time, plaques narrow arteries and restrict blood flow to the heart.
Above 240 mg/dL = high risk.

**MaxHR:** Maximum heart rate achieved during exercise stress test.
Healthy hearts can go higher. A low MaxHR for your age suggests poor cardiovascular fitness.
Expected maximum = 220 - Age. If your actual MaxHR is much lower, that's a concern.

**Oldpeak (ST Depression):** During a stress test (exercise ECG), the ST segment of the
heart rhythm can drop below baseline. This suggests the heart muscle is getting insufficient
oxygen during exertion — a sign of coronary artery disease. Values > 2.0 are significant.

**ST_Slope:** The slope of the ST segment after exercise.
Upsloping = normal. Flat or downsloping = increased risk of significant coronary disease.

**ChestPainType:**
- Typical Angina: chest pain triggered by exertion, relieved by rest — classic cardiac symptom
- Atypical Angina: chest pain not fitting the classic pattern
- Non-Anginal Pain: chest pain clearly not from the heart
- Asymptomatic: no chest pain despite abnormal findings — dangerous because no warning

### Parkinson's voice features in plain English

**Why voice?** Parkinson's affects the motor control neurons that control the larynx
(voice box). This causes measurable changes in voice characteristics years before
other motor symptoms appear, making voice a useful early biomarker.

**Jitter:** How much does the fundamental frequency vary between consecutive cycles?
A healthy voice has very consistent frequency. Parkinson's causes micro-tremors that
make frequency vary more than normal. Higher jitter = more irregularity.

**Shimmer:** How much does the amplitude (loudness) vary between consecutive cycles?
Similar to jitter but for loudness. Parkinson's causes amplitude instability.
Higher shimmer = more loudness variation.

**HNR (Harmonic-to-Noise Ratio):** Ratio of periodic (harmonic) signal to noise.
A healthy voice is mostly harmonic. Parkinson's introduces more noise/aperiodicity.
Lower HNR = noisier, less healthy voice.

**NHR (Noise-to-Harmonic Ratio):** Inverse of HNR. Higher NHR = worse.

**RPDE:** Measures the complexity/regularity of the voice signal.
A healthy voice is more predictable. Parkinson's voices are more complex/irregular.

**DFA (Detrended Fluctuation Analysis):** Measures long-range correlations in the voice signal.
Captures fractal-like self-similarity properties of the vocal signal.

**PPE (Pitch Period Entropy):** Entropy of fundamental frequency periods.
Higher entropy = more irregular pitch = more Parkinson's-like.

---

## 7. Observation Images Explained

The following images are generated by the training notebooks and stored in the models/ folder.

### Heart Disease Images (models/heart/)

**model_comparison.png**
Bar chart comparing accuracy of all 17 models trained in the notebook.
Blue = bagging methods (ExtraTrees, RF, Bagging)
Orange = boosting methods (AdaBoost, GradientBoosting)
Green = ensemble combinations (Voting, Stacking)
Gray = individual models (LR, SVM, KNN, NaiveBayes)
ExtraTrees is the tallest bar on the right.

**shap_beeswarm.png**
Each dot = one patient in the test set.
Each row = one feature.
Red dots (pushed right) = high feature value increases risk.
Blue dots (pushed left) = low feature value increases risk.
The wider the spread, the more that feature matters overall.
Key observation: ChestPainType and MaxHR are typically the top features.

**shap_bar.png**
Mean absolute SHAP values = global feature importance.
Shows which features matter most across ALL patients (not just one).
Unlike standard feature importance, SHAP importance accounts for feature interactions.

**shap_waterfall.png**
SHAP explanation for one specific patient.
Shows how each feature pushed the prediction away from the baseline (average) prediction.
The base value is what the model would predict with no information.

**shap_dependence.png**
Scatter plots for top 3 features showing:
X-axis: feature value
Y-axis: SHAP value for that feature
Color: another feature (interaction coloring)
Reveals non-linear relationships and interaction effects.

**shap_decision.png**
Shows how 50 test patients' predictions were built up, one feature at a time.
Lines start at the base value and move right or left with each feature.
Highlighted line = one specific patient. Shows the "path" to the final prediction.

**shap_heatmap.png**
Rows = patients sorted by predicted risk score.
Columns = features.
Red cells = feature increases this patient's risk.
Blue cells = feature decreases this patient's risk.
Good for spotting patterns across the whole test set.

**shap_feature_correlation.png**
Correlation matrix of SHAP values across features.
Two features that are highly correlated in SHAP space have interacting effects on predictions.
Useful for understanding feature dependencies captured by the model.

**shap_vs_lime_comparison.png**
Side-by-side SHAP and LIME for one patient.
Lets you see whether both methods agree on which features matter.
Agreement = more confident explanation. Disagreement = worth investigating.

**shap_kernel_svm.png**
SHAP explanation computed for the SVM model using KernelExplainer.
Slower and approximate (unlike TreeExplainer) but model-agnostic.
Used to show that SHAP can explain any model, not just tree-based ones.

**shap_risk_stratification.png**
Histogram of cumulative positive SHAP values (risk scores) by class.
Two distributions: healthy vs heart disease patients.
The better the model, the more separated these distributions are.

**lime_true_positive.png / lime_false_positive.png / lime_true_negative.png / lime_false_negative.png**
LIME explanations for four specific cases:
- True Positive: correctly predicted heart disease
- True Negative: correctly predicted healthy
- False Positive: wrongly predicted heart disease (was actually healthy)
- False Negative: missed a heart disease case
Comparing these reveals what features confuse the model.

### Parkinson's Images (models/parkinsons/)

**eda_target_distribution.png**
Bar and pie charts showing class balance.
195 patients: ~147 Parkinson's (75.4%), ~48 healthy (24.6%).
Important: significant class imbalance. Models must handle this.

**eda_distributions.png**
Histograms of first 16 features split by class.
Green = healthy, Red = Parkinson's.
Shows which features have clearly separated distributions between classes.
PPE and spread1 typically show the clearest separation.

**eda_correlation.png**
Full correlation heatmap of all features.
Many voice features are highly correlated (e.g. all jitter measures correlate with each other).
This is why MI-based feature selection is important — it picks the most informative ones
without too much redundancy.

**eda_target_correlation.png**
Absolute correlation of each feature with the target (0=healthy, 1=Parkinson's).
Red bars = high correlation (> 0.4) — strongest predictors.
Orange bars = medium correlation.
Blue bars = weak correlation.

**eda_boxplots.png**
Boxplots for top 12 features by target correlation.
For each feature, shows the distribution for healthy vs Parkinson's patients.
Clear separation = useful feature. Overlapping = less informative.

**feature_mutual_info.png**
Mutual Information scores for all features after engineering.
Red bars = top 25% (highest MI) — definitely selected.
Orange bars = middle 50% — selected.
Blue bars = bottom 25% (lowest MI) — dropped.
This determines which features make it into the model.

**model_comparison.png**
Bar chart comparing all models: baseline + tuned + ensembles.
Shows why the best model was chosen (highest test accuracy AND AUC).

**shap_beeswarm.png / shap_bar.png**
Same interpretation as heart disease images.
For Parkinson's, PPE, spread1, and MDVP:Fo(Hz) are typically top features.

**shap_waterfall_tp.png / shap_waterfall_tn.png**
SHAP waterfall for a True Positive (correctly detected Parkinson's) patient
and a True Negative (correctly identified healthy) patient.
Comparing both shows what values drive correct predictions in each direction.

**shap_dependence.png**
SHAP dependence plots for top 3 voice features.
Reveals how the model's understanding of each feature is non-linear.

**shap_decision.png**
Decision paths for first 60 test patients.
Shows diversity in how the model reaches its conclusions.

**shap_heatmap.png**
All test patients sorted by Parkinson's risk score.
Red = higher risk, Blue = lower risk.
Healthy patients cluster at the top, Parkinson's at the bottom.

**shap_feature_corr.png**
Correlation of SHAP values across features.
High SHAP correlation = these features capture similar information for the model.

**shap_risk_stratification.png**
Histogram of SHAP-based risk scores split by true label.
The less overlap, the better the model discriminates.

**lime_tp.png / lime_fp.png / lime_tn.png / lime_fn.png**
LIME explanations for the four error/success cases.
False negatives (missed Parkinson's) are the most clinically dangerous —
these images help understand what values fool the model.

**roc_confusion.png**
Confusion matrix and ROC curves for multiple models.
ROC AUC closer to 1.0 = better discrimination.
The best model curve should be furthest from the diagonal.

---

## 8. Application Features

### Login / Register
- SHA-256 hashed passwords (never stored in plain text)
- Each user's predictions and chats are isolated
- Session persists until logout

### Prediction Pages (Diabetes / Heart / Parkinson's)
- Input form with clinical reference ranges
- SHAP Waterfall, SHAP Importance, LIME tabs
- Risk level: Low (<30%) / Moderate (30-50%) / High (50-70%) / Very High (>70%)
- PDF report download after every prediction
- Results auto-saved to history

### Bulk CSV Upload
- Download template with correct column names
- Upload CSV of multiple patients
- Predictions run row by row with progress bar
- Summary metrics + downloadable results CSV
- Optional: save all to history

### OCR Report Upload
- Supports PNG, JPG, PDF, TIFF images
- Tesseract OCR extracts text
- Regex patterns detect health values (Glucose, HbA1c, BMI, BP, Cholesterol, HR)
- Groq AI analyses the full report text automatically
- Analysis saved as a 'report' chat session
- Extracted values can autofill Diabetes or Heart forms

### History & Tracker
- All predictions shown per logged-in user
- Filter by disease, limit records
- Plotly line chart of risk % over time
- Risk level pie chart, disease bar chart
- Export as CSV, delete by ID

### Recommendations Engine
Tab 1 - Evidence-Based Tips (instant, no API):
  Checks your actual feature values against clinical thresholds.
  Produces personalised tips that reference your specific numbers.

Tab 2 - AI-Generated Plan (Groq Llama 3):
  Your feature values + top SHAP drivers are sent to Llama 3.
  The AI generates a full Diet / Exercise / Lifestyle plan
  that explains WHY each SHAP-identified feature matters for you.

### AI Health Chatbot
- Multi-session: each conversation saved separately in SQLite
- Auto-titles sessions from first message
- Context-aware: system prompt includes your latest prediction + SHAP drivers
- Quick question buttons for common queries
- Session types: normal (💬) and report (📝, from OCR)
- Powered by Groq Llama 3.1 8b Instant (free, 14,400 requests/day)

### PDF Export
- Generated after every prediction
- Includes: risk banner, metrics row, feature values table, SHAP drivers table
- Font-safe: uses plain hyphens (not em-dashes) for fpdf2 compatibility
- Downloaded instantly as a .pdf file

---

## 9. Database Design

File: medixai_history.db (SQLite, auto-created)

### users table
```
id            INTEGER  primary key
username      TEXT     unique
password_hash TEXT     SHA-256 hash
created_at    TEXT     datetime string
```

### predictions table
```
id          INTEGER  primary key
user_id     INTEGER  foreign key -> users.id
ts          TEXT     timestamp
disease     TEXT     "Diabetes" | "Heart Disease" | "Parkinson's Disease"
prediction  INTEGER  0 = negative, 1 = positive
risk_pct    REAL     0.0 to 100.0
risk_level  TEXT     "Low Risk" | "Moderate Risk" | "High Risk" | "Very High Risk"
confidence  REAL     0.0 to 100.0
features    TEXT     JSON string of feature key->value pairs
shap_values TEXT     JSON string of feature key->SHAP value pairs
```

### chat_sessions table
```
id         INTEGER  primary key
user_id    INTEGER  foreign key -> users.id
title      TEXT     auto-generated from first message
chat_type  TEXT     "normal" | "report"
messages   TEXT     JSON array of {role, content} objects
created_at TEXT     datetime
updated_at TEXT     datetime (updated on each new message)
```

To completely reset: delete medixai_history.db and restart the app.

---

## 10. How to Add or Modify Things

### Add a new disease model
1. Train your model and save it as a .pkl file
2. Add a new file pages/your_disease.py following the pattern in pages/diabetes.py
3. Add feature definitions to utils/models.py
4. Add a load_your_disease() function in utils/models.py
5. Add the page to app.py sidebar radio and routing

### Change risk thresholds
Edit the classify_risk() function in utils/models.py:
```python
def classify_risk(pct):
    if pct < 30:   return "Low Risk", ...
    if pct < 50:   return "Moderate Risk", ...
```

### Add new recommendation rules
Edit pages/recommendations.py, add entries to _DIAB_RULES, _HEART_RULES, or _PARK_RULES:
```python
("feature_key", lambda v: v > threshold, "diet",
 "Your tip text here with {v:.1f} for the actual value."),
```

### Change the LLM model
Edit utils/llm.py, change:
```python
GROQ_MODEL = "llama-3.1-8b-instant"
```
Other free Groq models: llama-3.1-70b-versatile, mixtral-8x7b-32768

### Add PostgreSQL instead of SQLite
Replace utils/database.py with a psycopg2-based version.
All function signatures stay the same — only the internal DB calls change.
(The mdps_public.py file in this project shows the PostgreSQL version.)

---

## 11. Known Limitations

1. LIME for Heart: zeros used as background (no heart_lime_bg.npy).
   Fix: run np.save("heart_lime_bg.npy", X_train.values) in heart notebook
   and copy to models/heart/saved_models/

2. LIME for Parkinson's: same issue. Run np.save in the notebook.

3. OCR accuracy depends on image quality. Works best with:
   - Printed (not handwritten) text
   - High resolution (300+ DPI)
   - Horizontal text (not rotated)
   - Clear contrast between text and background

4. Parkinson's model feature list is determined at training time.
   If you retrain, the feature_names.json will change.

5. The database is local SQLite — suitable for personal/demo use.
   For multi-user deployment, migrate to PostgreSQL (see mdps_public.py).

6. fpdf2 built-in fonts do not support em-dashes, emoji, or non-Latin characters.
   Always use plain ASCII text in PDF generation.

---

## 12. Disclaimer

This application is built for EDUCATIONAL and RESEARCH purposes ONLY.

It is NOT a medical device. It is NOT certified for clinical use.
Predictions are based on statistical patterns in training data and do NOT
constitute medical advice, diagnosis, or treatment recommendations.

ALWAYS consult a qualified healthcare professional for any health concerns.
Never make medical decisions based solely on the output of this application.

The models may not generalise to populations different from their training data.
Parkinson's model: trained on a very small dataset (195 patients).
Results should be interpreted with appropriate caution.
