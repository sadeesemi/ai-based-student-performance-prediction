# 🎓 AI-Based Student Performance Prediction System

An AI-based system designed to **predict student academic risk and provide personalized learning recommendations** based on academic performance and learning behavior.

## ✨ Project Overview

The system analyzes student data to:

* 📊 Profile student performance
* ⚠️ Predict academic risk levels
* 🎯 Recommend personalized learning resources
* 📈 Help identify students who may need additional support

The project contains three main modules:

1. **Student Profiling**
2. **Risk Prediction**
3. **Personalized Intervention Recommendation**

## 🛠️ Technologies

* Python
* Flask
* Scikit-learn
* React
* Node.js
* MongoDB

## 📁 Project Structure

```text
research_code/
│
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   └── recommendation_module/
│
├── frontend/
│   ├── package.json
│   └── src/
│
├── .gitignore
└── README.md
```

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/sadeesemi/ai-based-student-performance-prediction.git
cd ai-based-student-performance-prediction
```

### 2. Run the Backend

Open a terminal:

```bash
cd backend
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the Flask backend:

```bash
python app.py
```

Backend:

```text
http://localhost:5000
```

### 3. Run the Recommendation Module

Open another terminal:

```bash
cd backend/recommendation_module
```

Run:

```bash
python main.py
```

### 4. Run the Frontend

Open another terminal:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the React application:

```bash
npm start
```

Frontend:

```text
http://localhost:3000
```

## 🎯 Personalized Recommendation Module

The recommendation module provides personalized learning resources based on:

* Student risk level
* Academic performance
* Learning behavior
* Weak areas
* Learning preferences

It combines **content-based recommendation, rule-based techniques, and knowledge-graph-based information** to generate suitable interventions.

## 👩‍💻 Academic Project

**AI-Based Student Performance Prediction System**
University of Moratuwa

---

⭐ If you find this project useful, feel free to explore the repository.
