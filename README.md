# 🏭 Nassau Candy AI Control Tower
### Factory Reallocation & Shipping Optimization System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](https://mattanikhil-nassau-candy-optimization-app-ue7miu.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Live Demo
**Interact with the AI Control Tower here:** [Nassau Candy Dashboard](https://mattanikhil-nassau-candy-optimization-app-ue7miu.streamlit.app/)

---

## 📌 Project Overview
Nassau Candy Distributor faced a common supply chain challenge: **static, legacy factory-product assignments.** Products were assigned to factories based on historical rules rather than data-driven efficiency. This led to:
* **Suboptimal shipping distances** and high lead times.
* **Margin erosion** due to logistics inefficiencies.
* **Lack of simulation tools** to quantify the impact of factory changes.

This project introduces **Decision Intelligence** by combining predictive machine learning and unsupervised clustering to recommend optimal factory reallocations.



---

## 🛠️ Technical Methodology

### 1. Data Preparation & Engineering
* **Lead Time Calculation:** Derived `Lead Time` by calculating the difference between `Order Date` and `Ship Date`.
* **Outlier Removal:** Used the **Interquartile Range (IQR)** method to remove statistically extreme lead times that could skew the model.
* **Feature Scaling:** Applied `StandardScaler` to numerical features (Sales, Units, Cost) to ensure they share a common variance.

### 2. Predictive Modeling (The "What-If" Engine)
I evaluated three models to predict expected shipping outcomes:
* **Linear Regression:** Used as a baseline.
* **Random Forest Regressor:** For handling non-linear relationships.
* **Gradient Boosting Regressor:** Selected as the **Best Performing Model** due to its superior RMSE and R² scores.

### 3. Route & Product Clustering (Unsupervised Learning)
Using **K-Means Clustering**, I grouped supply chain routes into three distinct categories:
* **🔴 High Congestion:** Consistently slow routes requiring immediate attention.
* **🟡 Moderate Volume:** Stable routes with average performance.
* **🟢 Express / Healthy:** Highly efficient corridors.



### 4. Optimization & Recommendation Logic
The system generates a **Strategic Priority Score** for every possible factory reassignment. This score is a weighted balance (50/50) of:
1. **Lead Time Reduction:** How much faster can we deliver?
2. **Profit Impact Stability:** Is the change financially safe?

---

## 📊 Key Performance Indicators (KPIs)
| KPI | Description | Value |
| :--- | :--- | :--- |
| **Lead Time Reduction** | Average operational gain across recommendations | **~15-20%** |
| **Scenario Confidence** | Reliability based on Model R² Score | **High (75%+)** |
| **Recommendation Coverage** | Percentage of products with better alternatives | **60%** |

---

## 💻 How to Run Locally
1. **Clone the repo:**
   ```bash
   git clone [https://github.com/MattaNikhil/Nassau-Candy-Optimization.git](https://github.com/MattaNikhil/Nassau-Candy-Optimization.git)

2. Install Dependencies:
pip install -r requirements.txt

3. Launch the App:
streamlit run app.py

4.Gemini said
To make your GitHub README.md look professional and explain the project in detail, you should break it down into the "Business Problem," "The Technical Solution," and "The Results."

Here is a complete, high-quality template you can copy and paste into your README.md file in VS Code.

Markdown
# 🏭 Nassau Candy AI Control Tower
### Factory Reallocation & Shipping Optimization System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](https://mattanikhil-nassau-candy-optimization-app-ue7miu.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Live Demo
**Interact with the AI Control Tower here:** [Nassau Candy Dashboard](https://mattanikhil-nassau-candy-optimization-app-ue7miu.streamlit.app/)

---

## 📌 Project Overview
Nassau Candy Distributor faced a common supply chain challenge: **static, legacy factory-product assignments.** Products were assigned to factories based on historical rules rather than data-driven efficiency. This led to:
* **Suboptimal shipping distances** and high lead times.
* **Margin erosion** due to logistics inefficiencies.
* **Lack of simulation tools** to quantify the impact of factory changes.

This project introduces **Decision Intelligence** by combining predictive machine learning and unsupervised clustering to recommend optimal factory reallocations.



---

## 🛠️ Technical Methodology

### 1. Data Preparation & Engineering
* **Lead Time Calculation:** Derived `Lead Time` by calculating the difference between `Order Date` and `Ship Date`.
* **Outlier Removal:** Used the **Interquartile Range (IQR)** method to remove statistically extreme lead times that could skew the model.
* **Feature Scaling:** Applied `StandardScaler` to numerical features (Sales, Units, Cost) to ensure they share a common variance.

### 2. Predictive Modeling (The "What-If" Engine)
I evaluated three models to predict expected shipping outcomes:
* **Linear Regression:** Used as a baseline.
* **Random Forest Regressor:** For handling non-linear relationships.
* **Gradient Boosting Regressor:** Selected as the **Best Performing Model** due to its superior RMSE and R² scores.

### 3. Route & Product Clustering (Unsupervised Learning)
Using **K-Means Clustering**, I grouped supply chain routes into three distinct categories:
* **🔴 High Congestion:** Consistently slow routes requiring immediate attention.
* **🟡 Moderate Volume:** Stable routes with average performance.
* **🟢 Express / Healthy:** Highly efficient corridors.



### 4. Optimization & Recommendation Logic
The system generates a **Strategic Priority Score** for every possible factory reassignment. This score is a weighted balance (50/50) of:
1. **Lead Time Reduction:** How much faster can we deliver?
2. **Profit Impact Stability:** Is the change financially safe?

---

## 📊 Key Performance Indicators (KPIs)
| KPI | Description | Value |
| :--- | :--- | :--- |
| **Lead Time Reduction** | Average operational gain across recommendations | **~15-20%** |
| **Scenario Confidence** | Reliability based on Model R² Score | **High (75%+)** |
| **Recommendation Coverage** | Percentage of products with better alternatives | **60%** |

---

## 💻 How to Run Locally
1. **Clone the repo:**
   ```bash
   git clone [https://github.com/MattaNikhil/Nassau-Candy-Optimization.git](https://github.com/MattaNikhil/Nassau-Candy-Optimization.git)
Install Dependencies:

Bash
pip install -r requirements.txt
Launch the App:

Bash
streamlit run app.py


👨‍💻 Project Credits
Lead Data Analyst: M Nikhil

Project Mentor: Sai Prasad Kagne

Organization: Unified Mentor Private Limited

### How to update this on GitHub:
1.  Open your `README.md` file in **VS Code**.
2.  **Select all** and delete the old text.
3.  **Paste** the code I wrote above.
4.  **Save** the file.
5.  Run these commands in your **Terminal**:
    ```powershell
    git add README.md
    git commit -m "Update professional project description."
    git push origin main
    ```

**Now your GitHub page looks like a professional Data Scientist's portfolio! Would you like me to help you take screenshots of your app to add to the "Visuals" section?**





























































































