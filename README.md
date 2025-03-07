# Gold Price Prediction using Artificial Neural Networks (ANNs)

## 📌 Introduction
Gold is one of the most valuable commodities and plays a crucial role in the global economy. Predicting its price is essential for investors, financial institutions, and economists. This project applies **Artificial Neural Networks (ANNs)** to predict gold prices using historical data. By leveraging deep learning techniques, we aim to develop an accurate model that identifies trends and patterns in the gold market.

## 📑 Project Overview
### Objectives:
- Analyze historical gold price data.
- Preprocess the data to remove inconsistencies and scale values.
- Train an ANN model to predict future gold prices.
- Evaluate the model's performance and identify areas for improvement.

### Key Features:
- Uses **Nasdaq** as a data source for reliable gold price information.
- Employs **MinMaxScaler** for data normalization.
- Implements a **Sequential Neural Network** with multiple **Dense layers**.
- Compares model performance across different architectures.

---

## 📂 Project Structure
The project is organized as follows:
```
├── data/                      # Raw and processed data files
│   ├── gold.csv               # Original dataset
│   ├── cleaned_gold.csv        # Processed dataset
│
├── models/                    # Trained models stored here
│   ├── ann_model.h5           # Trained ANN model
│
├── src/                       # Source code for data processing and model training
│   ├── data_preprocessing.py  # Data cleaning & feature engineering
│   ├── model_training.py      # ANN model training
│   ├── evaluation.py          # Model evaluation & performance metrics
│
├── results/                   # Stores evaluation metrics, graphs, and predictions
│   ├── performance_report.txt # Summary of model performance
│   ├── prediction_graphs.png  # Graphical representation of model predictions
│
├── README.md                  # Project documentation
```

---

## ⚙️ Installation
To run this project, install the required dependencies using the command below:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras
```
If you want to replicate the exact environment, consider using a virtual environment:
```bash
python -m venv gold_env
source gold_env/bin/activate  # On Mac/Linux
# OR
gold_env\Scripts\activate  # On Windows
pip install -r requirements.txt
```

---

## 📊 Dataset Information
The dataset consists of historical gold price data obtained from [Nasdaq](https://www.nasdaq.com/market-activity/commodities/gc%3Acmx). 
### Features in the dataset:
- **Date**: The date on which the price was recorded.
- **Open**: The opening price of gold on that day.
- **High**: The highest recorded price.
- **Low**: The lowest recorded price.
- **Close**: The closing price of gold.
- **Volume**: The total trading volume.

Preprocessing steps include:
- Handling missing values.
- Converting dates into numerical features.
- Normalizing price values using MinMaxScaler.

---

## 🚀 How to Use

### 1️⃣ Load and Preprocess Data
```python
from src.data_preprocessing import load_and_clean_data
df = load_and_clean_data("data/gold.csv")
```

### 2️⃣ Train the Model
```python
from src.model_training import train_ann
model = train_ann(df, epochs=50, batch_size=32)
```

### 3️⃣ Evaluate Model Performance
```python
from src.evaluation import evaluate_model
evaluate_model(model, df)
```

### 4️⃣ Make Predictions
```python
future_prices = model.predict(df)
print(future_prices)
```

---

## 📊 Model Performance
| Model          | Layers | Activation Function | Accuracy |
|---------------|--------|--------------------|----------|
| ANN (3 layers)| 3      | ReLU, Sigmoid      | 85%      |
| ANN (5 layers)| 5      | ReLU, Tanh, Sigmoid| 89%      |

The model was evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) to measure prediction accuracy.

---

## 📈 Results & Visualizations
To better understand model performance, several visualizations were generated:
- **Price Trend Analysis**: Displays how gold prices fluctuate over time.
- **Predicted vs. Actual Prices**: A graph comparing real prices with model predictions.
- **Loss Curves**: Illustrates how model loss decreases over epochs.

You can find these visualizations in the `results/` folder.

---

## 🔥 Future Improvements
- **Enhancing feature engineering** by including more economic indicators such as stock market data, inflation rates, and currency exchange rates.
- **Using more advanced architectures** such as **Long Short-Term Memory (LSTM)** networks for improved time series forecasting.
- **Optimizing hyperparameters** with grid search techniques to improve accuracy.
- **Real-time prediction API**: Deploying the model as a REST API for real-time gold price forecasting.

---

## 🤝 Contributors
📌 **Contributor:** Đặng Ngọc Hưng
📌 **Data Source:** Nasdaq Gold Market

If you find this project helpful, feel free to fork and contribute! 🚀

