
# 📚 Student Performance Prediction

This project aims to predict a student's exam performance based on various academic, personal, and socio-economic factors. Built using Python, trained with LightGBM, and deployed using Streamlit, the app provides an intuitive way to estimate scores and visualize important insights from the data.

---

## 🚀 Demo

You can run the app locally using:

```bash
streamlit run app.py
```

---

## 🧠 Problem Statement

Many students underperform due to various hidden factors. This project aims to:
- Predict student exam scores.
- Identify key features that influence performance.
- Provide a tool for educators to understand and support students better.

---

## 🛠️ Tech Stack

- **Python 3.12+**
- **LightGBM** – for model training
- **Pandas, NumPy** – for data handling
- **Matplotlib, Plotly** – for visualization
- **Streamlit** – for deployment
- **Joblib** – to save/load trained models
- **Scikit-learn** – for preprocessing

---

## 📊 Dataset

The dataset includes:
- Academic inputs (Hours Studied, Attendance, Previous Scores)
- Behavioral and socio-economic features (Motivation Level, Family Income, Gender, Internet Access, etc.)
- Output Label: `Exam_Score` (0–100)

---

## 🧪 Experiments

- **EDA**: Visualized feature distributions and correlation with scores.
- **Missing Data Handling**: Mode/mean imputation used where required.
- **Feature Encoding**: Label encoding for categorical features.
- **Model Training**: LightGBM was chosen for its accuracy and efficiency.
- **Model Accuracy**: Achieved ~92.6% R² score on test data.

---

## 🧩 App Features

- 🧮 Score Prediction from user input
- 📈 Feature importance visualization
- 📊 Exploratory data analysis with Plotly visuals

---

## 📂 Project Structure

```
student-performance-prediction/
│
├── app.py   # Streamlit app
├── \models
 └── model.pkl               
 └── scaler.pkl 
 └── model_training.ipynb                  
├──             # Fitted StandardScaler
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
└── StudentPerformanceFactors.csv  # Input dataset
```

---

## 🧑‍💻 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/abhaysankark/studentperformance_prediction.git
   cd studentperformance_prediction
   ```

2. Create virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   source venv/bin/activate  # macOS/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---
