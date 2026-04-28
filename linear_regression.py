import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

"""
Linear Regression Implementation following the CRISP-DM Framework.

Phases:
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment (Visualization)
"""

def main():
    # --- Phase 1: Business Understanding ---
    # Goal: Predict the value of 'y' based on 'x' using a linear relationship.
    # We expect a model of the form y = mx + c + noise.

    # --- Phase 2: Data Understanding ---
    print("Generating synthetic data...")
    np.random.seed(42)
    n_samples = 1000
    
    # Requirement: x is uniformly sampled from [-10, 10]
    X = np.random.uniform(-10, 10, n_samples).reshape(-1, 1)
    
    # Requirement: noise is drawn from a normal distribution N(0, 10)
    noise = np.random.normal(0, 10, n_samples).reshape(-1, 1)
    
    # Requirement: y = 10x + 30 + noise
    y = 10 * X + 30 + noise
    
    print(f"Generated {n_samples} data points.")

    # --- Phase 3: Data Preparation ---
    # Splitting data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Phase 4: Modeling ---
    # Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # --- Phase 5: Evaluation ---
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Model Evaluation ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    print(f"Coefficient (m): {model.coef_[0][0]:.4f}")
    print(f"Intercept (c): {model.intercept_[0]:.4f}")
    print(f"Learned Equation: y ~ {model.coef_[0][0]:.2f}x + {model.intercept_[0]:.2f}")

    # --- Phase 6: Deployment (Visualization) ---
    print("\nVisualizing results...")
    plt.figure(figsize=(10, 6))
    
    # Requirement: Scatter the original data points in blue
    plt.scatter(X, y, color='blue', alpha=0.3, label='Original Data', s=10)
    
    # Requirement: Draw the fitted regression line in red
    X_range = np.linspace(-10, 10, 100).reshape(-1, 1)
    y_range_pred = model.predict(X_range)
    plt.plot(X_range, y_range_pred, color='red', linewidth=3, label='Regression Line')
    
    plt.title('Linear Regression: CRISP-DM Framework', fontsize=14)
    plt.xlabel('x (Feature)', fontsize=12)
    plt.ylabel('y (Target)', fontsize=12)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot for reference
    plt.savefig('regression_plot.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved to regression_plot.png")

if __name__ == "__main__":
    main()
