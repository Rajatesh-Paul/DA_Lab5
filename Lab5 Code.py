import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
X_data = pd.read_csv('linearX.csv')
Y_data = pd.read_csv('linearY.csv')
X_data['0.99523'] = Y_data['0.99523']
def scale_min_max(column):
    min_val = column.min()
    max_val = column.max()
    return (column - min_val) / (max_val - min_val)
X_data['9.1'] = scale_min_max(X_data['9.1'])
X_data['0.99523'] = scale_min_max(X_data['0.99523'])
X_data.rename(columns={'0.99523': 'col2', '9.1': 'col1'}, inplace=True)
X = X_data.drop('col2', axis=1)
y = X_data['col2']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
class GradientDescent:
    def __init__(self, learning_rate, epochs):  # Corrected constructor method
        self.slope = 0
        self.intercept = 1
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.cost_history = []
    
    def fit(self, X_train, y_train):
        X_train = np.array(X_train).flatten()
        y_train = np.array(y_train)
        
        for epoch in range(self.epochs):
            y_pred = self.slope * X_train + self.intercept
            slope_gradient = (-2 / len(X_train)) * np.sum((y_train - y_pred) * X_train)
            intercept_gradient = (-2 / len(X_train)) * np.sum(y_train - y_pred)
            
            self.slope -= self.learning_rate * slope_gradient
            self.intercept -= self.learning_rate * intercept_gradient
            
            cost = (1 / len(X_train)) * np.sum((y_train - y_pred) ** 2)
            self.cost_history.append(cost)
        
        print("Intercept: ", self.intercept, "Slope: ", self.slope)
    
    def predict(self, X_test):
        X_test = np.array(X_test).flatten()
        return self.slope * X_test + self.intercept
gd_model = GradientDescent(learning_rate=0.05, epochs=1000)
gd_model.fit(X_train, y_train)
y_predictions = gd_model.predict(X_test)
print(y_predictions)
plt.figure(figsize=(10, 6))
plt.plot(range(1, 51), gd_model.cost_history[:50], marker='o', label='Cost vs Iterations')
plt.title("Cost Function vs Epochs (50)")
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.legend()
plt.grid()
plt.show()
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data Points')
X_line = np.linspace(X.min().values[0], X.max().values[0], 100)
y_line = gd_model.slope * X_line + gd_model.intercept
plt.plot(X_line, y_line, color='red', label='Fitted Line')
plt.title("Dataset and Fitted Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid()
plt.show()
learning_rates = [0.005, 0.5, 5]
learning_rate_results = {}
for lr in learning_rates:
    temp_gd_model = GradientDescent(learning_rate=lr, epochs=50)
    temp_gd_model.fit(X_train, y_train)
    learning_rate_results[lr] = temp_gd_model.cost_history
# Plot cost vs iterations for different learning rates
plt.figure(figsize=(10, 6))
for lr, cost_history in learning_rate_results.items():
    plt.plot(range(1, 51), cost_history, marker='o', label=f"Learning Rate = {lr}")
plt.title("Cost Function vs Epochs (Learning Rates: 0.005, 0.5, 5)")
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.legend()
plt.grid()
plt.show()
class MiniBatchGD(GradientDescent):
    def fit(self, X_train, y_train, batch_size=10):
        X_train = np.array(X_train).flatten()
        y_train = np.array(y_train)
        n = len(X_train)
        
        for epoch in range(self.epochs):
            for batch_start in range(0, n, batch_size):
                X_batch = X_train[batch_start:batch_start + batch_size]
                y_batch = y_train[batch_start:batch_start + batch_size]
                
                y_pred = self.slope * X_batch + self.intercept
                slope_grad = (-2 / len(X_batch)) * np.sum((y_batch - y_pred) * X_batch)
                intercept_grad = (-2 / len(X_batch)) * np.sum(y_batch - y_pred)
                
                self.slope -= self.learning_rate * slope_grad
                self.intercept -= self.learning_rate * intercept_grad
                
            cost = (1 / n) * np.sum((y_train - (self.slope * X_train + self.intercept)) ** 2)
            self.cost_history.append(cost)
mini_batch_gd = MiniBatchGD(learning_rate=0.5, epochs=50)
mini_batch_gd.fit(X_train, y_train, batch_size=10)
class StochasticGD(GradientDescent):
    def fit(self, X_train, y_train):
        X_train = np.array(X_train).flatten()
        y_train = np.array(y_train)
        n = len(X_train)
        
        for epoch in range(self.epochs):
            for i in range(n):
                X_single = X_train[i]
                y_single = y_train[i]
                
                y_pred = self.slope * X_single + self.intercept
                slope_grad = -2 * (y_single - y_pred) * X_single
                intercept_grad = -2 * (y_single - y_pred)
                
                self.slope -= self.learning_rate * slope_grad
                self.intercept -= self.learning_rate * intercept_grad
                
            cost = (1 / n) * np.sum((y_train - (self.slope * X_train + self.intercept)) ** 2)
            self.cost_history.append(cost)
stochastic_gd_model = StochasticGD(learning_rate=0.5, epochs=50)
stochastic_gd_model.fit(X_train, y_train)
plt.figure(figsize=(10, 6))
plt.plot(range(1, 51), stochastic_gd_model.cost_history[:50], marker='o', label='Stochastic GD Cost')
plt.title("Cost Function vs Epochs (Stochastic GD)")
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.legend()
plt.grid()
plt.show()
