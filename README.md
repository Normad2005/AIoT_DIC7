# Linear Regression CRISP-DM Explorer

An interactive web application built with **Streamlit** that demonstrates the complete **CRISP-DM** (Cross-Industry Standard Process for Data Mining) workflow for solving a linear regression problem.

## 🚀 Features

- **Interactive Modeling**: Adjust parameters like slope ($a$), intercept ($b$), noise level, and sample size using real-time sliders.
- **CRISP-DM Workflow**: Organized into phases:
    1. **Data Understanding**: Generation of synthetic data with controlled parameters.
    2. **Data Preparation**: Automatic train-test splitting.
    3. **Modeling & Evaluation**: Real-time training of a `scikit-learn` Linear Regression model with MSE and $R^2$ metrics.
    4. **Deployment**: Interactive **Altair** charts with zooming, panning, and tooltip functionality.
- **Code Highlights**:
    - `linear_regression.py`: A clean, standard Python implementation.
    - `app.py`: The interactive Streamlit application.
    - `chat.md`: A detailed record of the project development process.

## 🛠️ Installation

Ensure you have Python installed, then install the required dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib streamlit altair
```

## 📈 Usage

To launch the interactive dashboard:

```bash
python -m streamlit run app.py
```

## 📂 Project Structure

- `app.py`: Main Streamlit application.
- `linear_regression.py`: Core logic for data generation and modeling.
- `chat.md`: Project communication and requirement logs.
- `regression_plot.png`: Static visualization export.
