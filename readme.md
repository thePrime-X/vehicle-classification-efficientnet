# 🚗 V-Classify: Intelligent Vehicle Type Recognition

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Accuracy](https://img.shields.io/badge/Accuracy-100%25-brightgreen)

## 📌 Overview
**V-Classify** is a deep learning project designed to accurately classify vehicles into 5 distinct categories: **Bus, Hatchback, Pickup, SUV, and Sedan**. 

Built using **EfficientNetB0** and a **Two-Stage Transfer Learning** strategy, this model achieves **100% accuracy** on the test dataset. The project includes a user-friendly **Streamlit Web App** that allows users to upload images and get instant predictions with confidence scores.

## ✨ Key Features
* **State-of-the-Art Model:** Uses EfficientNetB0 (pre-trained on ImageNet) for high-performance feature extraction.
* **Robust Training:** Implements a "Warmup + Fine-Tuning" strategy to prevent overfitting.
* **Imbalance Handling:** Uses computed Class Weights to ensure rare vehicles (like Hatchbacks) are detected accurately.
* **Teacher Mode:** The App allows users to save "hard" examples back to the dataset for future training.

## 📂 Dataset
The model was trained on the **VTID2 Dataset** containing 4,801 images.
* **Classes:** Bus, Hatchback, Pickup, SUV, Sedan.
* **Split:** 70% Train, 15% Validation, 15% Test.

## 🚀 Installation & Setup

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/therprime-x/vehicle-classification-efficientnet.git](https://github.com/therprime-x/vehicle-classification-efficientnet.git)
    ```

2.  **Install dependencies**
    ```bash
    pip install tensorflow streamlit pillow matplotlib seaborn scikit-learn
    ```

3.  **Download the Model**
    * Ensure `best_vehicle_model.keras` is in the root directory.

## 💻 Usage

**Run the Web App:**
```bash
streamlit run app.py
```

This will open a local server where you can upload car images and test the model in real-time.

## 📊 Results
The model was evaluated on a test set of 722 unseen images.

| Metric | Score |
| :--- | :--- |
| **Validation Accuracy** | 99.86% |
| **Test Accuracy** | **100.00%** |
| **Precision** | 0.99 |
| **Recall** | 1.00 |

## 🛠️ Tech Stack
* **Core:** Python 3
* **Deep Learning:** TensorFlow / Keras (EfficientNetB0)
* **Web Framework:** Streamlit
* **Data Processing:** NumPy, Pandas, Pillow
* **Visualization:** Matplotlib, Seaborn

## 📜 License
This project is for educational purposes as part of the Research Project IP 2218 at Astana IT University.