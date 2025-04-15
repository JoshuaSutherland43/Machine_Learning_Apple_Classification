# 🍏 Apple Bruise Classification - Hackathon Winning Project 🏆

**By Joshua Sutherland & Samkelo Maswana**\
🗓️ *Hackathon Dates: April 9–10, 2025*

## 📌 Problem Statement

Manual sorting of apples for quality control is time-consuming, inconsistent, and labor-intensive.\
Bruises are often hard to detect, especially in early stages or when not visually prominent.\
There’s a pressing need for an **automated, efficient, and accurate model** for bruise detection across apple varieties.

## 💡 Our Solution

We developed a **Support Vector Machine (SVM)** model for classifying **bruised vs. sound apples** across three apple types:

- Granny Smith 🍏
- Royal Gala 🍎
- Golden Delicious 🍏

Each apple variety had its **own dedicated SVM model**, enabling specialized learning and tailored classification.

## ⚙️ Development & Testing Approach

### 🧪 Data Description

- **Source:** Infrared intensity data from apples
- **Features:** Wavenumber-transformed wavelength values
- **Labels:** "S" = Sound, "B" = Bruised
- **Format:** Rolling feature windows
- **Sample ID example:** `GD-ch-bruise1.5h-10a`

### 🧑‍🧬 Model Process

- **Feature Normalization:** StandardScaler
- **Target Transformation:** "S" ➔ 1, "B" ➔ 0
- **Train/Test Split:** 70% training, 30% testing
- **Feature Selection:** Rolling window-based filtering
- **Hyperparameter Tuning:** GridSearchCV
  - Parameters: C, kernel, gamma
- **CPU & Memory Optimization:** Sam handled tuning for efficient resource usage

### 🔭 Tools Used

- Python (Jupyter Notebook)
- Scikit-learn
- NumPy / Pandas / Matplotlib
- psutil (for CPU monitoring)

## 📊 Results & Model Comparison

| Criteria         | Our SVM Model                            | Hackathon Model (RFC + Logistic) |
| ---------------- | ---------------------------------------- | -------------------------------- |
| Model Type       | SVM (GS, GD, RG)                         | Random Forest + Logistic         |
| Accuracy         | GS: **0.8355** GD: **0.93** RG: **0.8343** | GS: 0.7947 GD: 0.8787 RG: 0.8284 |
| Precision        | GS: **0.8355** GD: **0.82** RG: **0.7642** | GS: 0.7647 GD: 0.8461 RG: 0.7777 |
| Interpretability | Moderate                                 | High                             |
| Complexity       | Medium                                   | Low                              |

## 🏆 Outcome

After long hours of development, debugging, and testing, **we placed 1st** in the competition!\
The experience was intense, challenging, but incredibly rewarding — a hard-earned victory!

## 💭 Reflection & Future Plans

As this was my **first deep dive into machine learning**, this challenge opened a new world of possibilities.\
I’m inspired to:

- **Pursue further learning in Data Science & ML**
- **Apply ML to mobile and web app development**
- **Integrate smart data-driven systems in real-world applications**
- **Continue collaborating on cutting-edge challenges**

## 🚀 How to Run the Project

### 🔧 Requirements

Install the required libraries using pip:

```bash
pip install numpy pandas scikit-learn matplotlib psutil
```

### 📁 Download & Setup

1. Clone the repository:

```bash
git clone https://github.com/your-username/apple-bruising-svm.git
cd apple-bruising-svm
```

2. Launch the Jupyter Notebook:

```bash
jupyter notebook
```

3. Open the notebook file (e.g., `apple_svm_model.ipynb`)
4. Follow the cells to run data preprocessing, model training, and evaluation.

## 👥 Team

- **Joshua Sutherland** – SVM Model Development, Model Integration, Implementation
- **Samkelo Maswana** – CPU Optimization, Code Streamlining

 
