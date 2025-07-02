
#Obesity‑Level Prediction – Ensemble Learning Project



Predict an individual’s obesity category (7 classes) from basic lifestyle and anthropometric inputs using an ensemble of classical machine‑learning models.This repository contains the notebook, Streamlit script, and trained models submitted as part of the course/club assignment.

📂 Repository layout

Path

Purpose

OBESITY_EnsembleLearning.ipynb

End‑to‑end exploration & model‑building notebook

model_description.py

Minimal Streamlit app that loads a pickled model ensemble and predicts

LogisticR (1).pkl

Logistic Regression (baseline)

SVM (1).pkl

Support‑Vector Machine

decisiontree (1).pkl

Decision Tree (baseline)

Randomforest (1).pkl

Random‑Forest classifier

SoftVoting.pkl

Soft‑voting ensemble (best‑probability aggregation)

Bagging.pkl, BaggedDT.pkl

Bagging meta‑estimator & its bagged decision tree

README.md

You are here

Note: Filenames contain "(1)" because they were exported from Google Colab; feel free to rename for production use.

🗂️ Dataset

Source: Obesity Levels data set (UCI ML Repository).

Target classes (7): Insufficient_Weight, Normal_Weight, Overweight_Level_I, Overweight_Level_II, Obesity_Type_I, Obesity_Type_II, Obesity_Type_III.

Features (16): Age, Gender, Height, Weight, and 12 lifestyle variables (FCVC, NCP, CAEC … MTRANS).

The notebook demonstrates cleaning, exploratory analysis, one‑hot/ordinal encoding, train‑test split (80/20 stratified) and cross‑validation.

⚙️ Installation

# 1 – clone the repo
$ git clone https://github.com/<user>/Obesity_EnsembleLearning.git
$ cd Obesity_EnsembleLearning

# 2 – create & activate env (conda or venv)
$ python -m venv .venv
$ source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3 – install deps
(.venv)$ pip install -r requirements.txt  # see below

Minimal requirements.txt

pandas
numpy
scikit-learn
streamlit
matplotlib
joblib

(Add jupyter & seaborn if you plan to run the notebook.)

🚀 Quick start

A) Notebook workflow

jupyter notebook OBESITY_EnsembleLearning.ipynb

Follow the cells to reproduce the preprocessing, training, and evaluation pipeline.

B) Streamlit demo

streamlit run model_description.py

A form UI appears in your browser → fill in the inputs → click Submit → see the predicted obesity category from each ensemble member.

Tip: If you relocate the *.pkl files, update the paths at the top of model_description.py.

📈 Results (hold‑out test)

Model

Accuracy

Notes

Decision Tree

86.77%

simple baseline

Logistic Regression

84.15%

robust, linear baseline

Support Vector Machine

79.48%

kernel-based, no ensemble

Random Forest

76.78%

bagged trees (raw)

Bagging

86.91%

best overall

Boosting

76.65%

underperforms here

Soft Voting

83.88%

average of probabilities

Hard Voting

48.91%

majority vote – poor

Weighted Voting

83.62%

weighted by validation scores

Confusion matrices & ROC curves are in the notebook.

🛠️ Future work

Hyper‑parameter optimisation with Optuna

Class‑imbalance handling (SMOTE‑ENN)

Calibrated probability estimates & expected‑cost framework

Save models with joblib + version metadata to avoid "Ran out of input" errors

Dockerfile for reproducible deployment

👤 Author

Yashveer SabharwalBITS Goa LinkedIn: www.linkedin.com/in/yashveer-sabharwal-034846250

