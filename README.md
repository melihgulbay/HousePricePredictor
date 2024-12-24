# Istanbul House Price Predictor

A desktop application that predicts house prices in Istanbul using machine learning models.

## Features

- Predicts house prices using both Linear Regression and Random Forest models
- Interactive GUI with detailed statistics
- Supports all Istanbul districts
- Provides comprehensive property insights:
  - Price trends per square meter
  - Room type distribution
  - Size statistics
  - Price ranges by property size

## Requirements

- Python 3.x
- Required packages:
  ```
  pip install pandas numpy scikit-learn tkinter
  ```

## Usage

1. Ensure you have a `house_prices.csv` file with the following columns:
   - Fiyat (Price)
   - m² (Brüt) (Gross Area)
   - Oda Sayısı (Room Type)
   - Bölge (District)

2. Run the application:
   ```
   python main.py
   ```

3. Input your parameters:
   - Select a prediction model
   - Choose a district
   - Enter the property area
   - Select room type
   - Click "Predict Price" to see results

## Models

- **Linear Regression**: Better for understanding price trends
- **Random Forest**: Generally more accurate for predictions

## File Structure

- `main.py`: Application entry point
- `gui.py`: GUI implementation
- `models.py`: Machine learning model training
- `house_prices.csv`: Dataset





