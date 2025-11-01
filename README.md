## Student Performance
- Dataset: https://archive.ics.uci.edu/dataset/320/student+performance
# Tasks
- Classifcation: Pass vs Failure in the Final Grade col 
- Regression: Predict the final grade G3

# Setup Instructions
```
# Clone the repositiory

git clone https://github.com/FalconCharge/Machine-Learning-Project.git
cd Machine-Learning-Project

# Install Dependices

pip install -r requirements.txt

# Run train_baselines.py

python src/train_baseline.py

#open mlflow to see the results
mlflow ui

```
In your browser goto http://127.0.0.1:5000


Metrics Used:
- Classification: Accuracy + F1-score, with confusion matrix
- Regression: MAE + RMSE, with residual analysis