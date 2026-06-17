# 📊 Customer Churn Prediction

An end-to-end deep learning web application that predicts whether a bank customer is likely to churn, built using an Artificial Neural Network (ANN) and deployed with Streamlit.

---

## 🚀 Live Demo

- 🔗 [Customer Churn Predictor](https://annclassification-mthomjbp2i8jjjea5uvcwg.streamlit.app/) 

---

## 🧠 About the Project

Customer churn is one of the biggest challenges in the banking industry. Losing a customer costs far more than retaining one. This project builds a binary classification model using an ANN that takes in customer details and outputs a **churn probability** along with a **risk level** (Low / Medium / High).

The model was trained on the classic [Bank Customer Churn Dataset](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction) and deployed as an interactive web app.

---

## ⚙️ How It Works

```
Raw Input -> Label Encoding (Gender) -> One-Hot Encoding (Geography)
         -> Standard Scaling -> ANN Model -> Churn Probability (0 to 1)
```

### ANN Architecture
- **Input Layer** : 12 features after encoding
- **Hidden Layer 1** : 64 neurons, ReLU activation
- **Hidden Layer 2** : 32 neurons, ReLU activation
- **Output Layer** : 1 neuron, Sigmoid activation (binary classification)

### Why ReLU?
ReLU (Rectified Linear Unit) outputs zero for negative values and keeps positive values as-is. It avoids the vanishing gradient problem that older activation functions like Sigmoid had in deep hidden layers - meaning weights update properly during backpropagation and the network actually learns.

### Why Sigmoid on output?
Since this is a binary classification task (churn or not), the output neuron needs to return a probability between 0 and 1. Sigmoid squishes any value into that range perfectly.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| TensorFlow / Keras | ANN model building & training |
| Scikit-learn | Preprocessing (encoding, scaling) |
| Pandas & NumPy | Data manipulation |
| Streamlit | Web app deployment |
| Matplotlib | Training visualization |

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── app.py                        # Streamlit web app
├── experiments.ipynb             # Model training & experimentation
├── prediction.ipynb              # Prediction pipeline walkthrough
├── requirements.txt              # Python dependencies
├── .gitignore                    # Ignored files
└── README.md                     # You are here
```

---


## 📊 Input Features

| Feature | Description |
|---|---|
| Credit Score | Customer's credit score (300–850) |
| Geography | Country — France, Germany, Spain |
| Gender | Male / Female |
| Age | Customer's age |
| Tenure | Years with the bank |
| Balance | Account balance |
| Number of Products | Bank products used (1–4) |
| Has Credit Card | Whether customer has a credit card |
| Is Active Member | Whether customer is an active member |
| Estimated Salary | Customer's estimated salary |

---

## 📈 Output

- **Churn Probability** - a score between 0 and 1
- **Risk Level:**
  - 🟢 Low Risk - probability < 0.50
  - 🟡 Medium Risk - probability 0.50 to 0.75
  - 🔴 High Risk - probability > 0.75

---

## 📚 What I Learned

- How ANNs work -> forward pass, loss calculation, backpropagation, weight updates
- Role of activation functions and the difference between ReLU (hidden layers) and Sigmoid (output layer)
- Full preprocessing pipeline -> Label Encoding, One-Hot Encoding, Standard Scaling
- Saving and loading model artifacts (`.h5`, `.pkl`) for deployment
- Building a production-aware Streamlit app with caching, error handling, and clean UI

---

## 🙋‍♂️ Author

**Hsrshvardhan**
- 🔗 [LinkedIn](https://www.linkedin.com/in/harsh-attri-b72ab2248/) 
- 🐙 [Github Repo](https://github.com/harshattri4586/Ann_Classification)

---