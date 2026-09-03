AI-Based Student Performance Prediction System

An AI-based system that analyzes student academic and learning behavior data to predict performance risk and provide personalized intervention recommendations.

Project Overview

The system consists of three main modules:

Module 01 – Student Profiling
Module 02 – Risk Prediction
Module 03 – Personalized Intervention Recommendation

This repository contains the complete implementation of the system, including the backend, frontend, machine learning models, recommendation module, and generated outputs.

Technologies
Python
Flask
Scikit-learn
NumPy
React
Node.js
MongoDB
Project Structure
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
How to Run the Project
1. Run the Backend / Recommendation Module

Open a terminal in the project folder:

cd backend

Create a Python virtual environment:

python -m venv .venv
Windows
.venv\Scripts\activate
macOS / Linux
source .venv/bin/activate

Install the required Python packages:

pip install -r requirements.txt

Go to the recommendation module:

cd recommendation_module

Run the training and recommendation pipeline:

python main.py

This generates the required recommendation models, evaluation results, reports, and dashboard data.

2. Run the Frontend

Open a new terminal and go to the frontend folder:

cd frontend

Install the dependencies:

npm install

Start the React application:

npm start

The application will open at:

http://localhost:3000
3. Optional Flask API

To run the backend API:

cd backend
python app.py

The API will be available at:

http://localhost:5000
4. Test a Recommendation

You can test the recommendation module for a student using:

cd backend/recommendation_module
python recommend_student.py ST1008 --verify

Replace ST1008 with another student ID when required.

Module 03

The Personalized Intervention Recommendation module combines:

Rule-based filtering
Content-based recommendation using TF-IDF
Knowledge Graph and GraphSAGE-based features
Random Forest ranking
Explainable AI techniques
Risk and behavior-based intervention recommendations

The dashboard allows users to search for a student and view their risk information, learning needs, recommendations, and intervention suggestions.

Academic Project

AI-Based Student Performance Prediction System
University of Moratuwa