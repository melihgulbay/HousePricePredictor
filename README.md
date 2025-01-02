Here’s your GitHub README optimized and polished for presentation: 


# Istanbul House Price Predictor

A machine learning application designed to predict house prices in Istanbul, featuring an interactive GUI, advanced visualizations, and multiple robust prediction models.

## Features

### Prediction Models
- **Linear Regression**
- **Random Forest**
- **Support Vector Regression (SVR)**
- **Gradient Boosting**
- **XGBoost**

### Interactive Visualizations
- District-wise price trends
- Price vs. area analysis
- Heatmaps for district-based pricing
- Room type distribution
- Box plots for price variation
- Area distribution analysis
- Interactive price map
- Price per square meter analysis
- Room size distribution
- District size distribution

### Model Performance Metrics
- Cross-validation scores
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- R² scores
- Prediction standard deviation



## Installation

### Step 1: Clone the Repository
git clone https://github.com/melihgulbay/HousePricePredictor.git
cd HousePricePredictor


### Step 2: Install Dependencies

pip install -r requirements.txt


### Step 3: Launch the Application

python main.py




## Requirements

- Python 3.8+
- Libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `folium`, `geopandas`, `tkinter`



## Usage

1. Launch the application with:
   python main.py
   
2. The application will train the models using the provided dataset.

3. Once training is complete, the GUI will launch with three main sections:
   - **Price Prediction**: Input property details to get a predicted price.
   - **Visualizations**: Explore comprehensive data visualizations.
   - **Model Metrics**: Compare the performance of all implemented models.



## Data

The application relies on a dataset of Istanbul house prices (`house_prices.csv`) with the following features:
- **Area (m²)**: Size of the property in square meters.
- **Number of Rooms**: Total rooms in the property.
- **District**: Geographic district of the property.
- **Price (TL)**: Property price in Turkish Lira.


## Project Structure


HousePricePredictor/
├── main.py                  # Application entry point
├── models.py                # ML model definitions and training
├── gui.py                   # Main GUI implementation
├── gui_prediction.py        # Prediction interface
├── gui_visualization.py     # Data visualization
├── gui_metrics.py           # Model metrics display
├── price_map_visualizer.py  # Interactive map visualization
├── house_prices.csv         # Dataset
└── requirements.txt         # Project dependencies








This format ensures clarity, readability, and accessibility for your project's users and contributors. Let me know if you'd like any further adjustments!
