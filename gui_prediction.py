import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np

class GUIPrediction:
    def __init__(self, prediction_frame, models_data, df):
        self.models_data = models_data
        self.models = {k: v for k, v in models_data.items() if k != 'metrics'}
        self.df = df
        
        # Define available districts and room types first
        self.districts = [
            'Adalar', 'Arnavutkoy', 'Atasehir', 'Avcilar', 'Bagcilar', 
            'Bahcelievler', 'Bakirkoy', 'Basaksehir', 'Bayrampasa', 'Besiktas', 
            'Beykoz', 'Beylikduzu', 'Beyoglu', 'Buyukcekmece', 'Catalca', 
            'Cekmekoy', 'Esenler', 'Esenyurt', 'Eyupsultan', 'Fatih', 
            'Gaziosmanpasa', 'Gungoren', 'Kadikoy', 'Kagithane', 'Kartal', 
            'Kucukcekmece', 'Maltepe', 'Pendik', 'Sancaktepe', 'Sariyer', 
            'Silivri', 'Sultanbeyli', 'Sultangazi', 'Sile', 'Sisli', 
            'Tuzla', 'Umraniye', 'Uskudar', 'Zeytinburnu'
        ]
        
        self.room_types = [
            'Stüdyo (1+0)', '1+1', '1.5+1', '2+0', '2+1', '2.5+1', '2+2',
            '3+0', '3+1', '3.5+1', '3+2', '3+3', '4+0', '4+1', '4.5+1',
            '4.5+2', '4+2', '4+3', '4+4', '5+1', '5.5+1', '5+2', '5+3',
            '5+4', '6+1', '6+2', '6.5+1', '6+3', '6+4', '7+1', '7+2',
            '7+3', '8+1', '8+2', '8+3', '8+4', '9+1', '9+2', '9+3',
            '9+4', '9+5', '9+6', '10+1', '10+2'
        ]
        
        # Create left and right frames
        self.left_frame = ttk.LabelFrame(prediction_frame, text="Input Parameters", padding="20")
        self.left_frame.grid(row=0, column=0, sticky=(tk.N, tk.W, tk.E, tk.S), padx=10)
        
        self.right_frame = ttk.LabelFrame(prediction_frame, text="Results", padding="20")
        self.right_frame.grid(row=0, column=1, sticky=(tk.N, tk.W, tk.E, tk.S), padx=10)

        self.setup_input_frame()
        self.setup_results_frame()

    def setup_input_frame(self):
        # Input elements in left frame
        ttk.Label(self.left_frame, text="Model:", style='Modern.TLabel').grid(row=0, column=0, padx=15, pady=10, sticky=tk.W)
        self.model_choice = ttk.Combobox(
            self.left_frame, 
            values=[
                'Linear Regression',
                'Random Forest',
                'Support Vector Regression',
                'Gradient Boosting',
                'XGBoost'
            ],
            width=30, 
            style='Modern.TCombobox'
        )
        self.model_choice.grid(row=0, column=1, padx=15, pady=10)
        self.model_choice.set('Random Forest')
        
        ttk.Label(self.left_frame, text="District:", style='Modern.TLabel').grid(row=1, column=0, padx=15, pady=10, sticky=tk.W)
        self.bolge = ttk.Combobox(self.left_frame, values=self.districts, width=30, style='Modern.TCombobox')
        self.bolge.grid(row=1, column=1, padx=15, pady=10)
        
        ttk.Label(self.left_frame, text="Area (m²):", style='Modern.TLabel').grid(row=2, column=0, padx=15, pady=10, sticky=tk.W)
        self.area = ttk.Entry(self.left_frame, width=32, font=('Helvetica', 11))
        self.area.grid(row=2, column=1, padx=15, pady=10)
        
        ttk.Label(self.left_frame, text="Room Type:", style='Modern.TLabel').grid(row=3, column=0, padx=15, pady=10, sticky=tk.W)
        self.rooms = ttk.Combobox(self.left_frame, values=self.room_types, width=30, style='Modern.TCombobox')
        self.rooms.grid(row=3, column=1, padx=15, pady=10)
        
        predict_button = ttk.Button(self.left_frame, text="Predict Price", 
                                  command=self.predict_price, style='Modern.TButton')
        predict_button.grid(row=4, column=0, columnspan=2, pady=20)

    def setup_results_frame(self):
        self.result_label = ttk.Label(self.right_frame, style='Result.TLabel', 
                                    background='#ffffff', relief='solid')
        self.result_label.grid(row=0, column=0, pady=10, sticky=(tk.W, tk.E))
        
        self.stats_text = tk.Text(self.right_frame, wrap=tk.WORD, height=30,
                                font=('Helvetica', 11), relief='solid')
        self.stats_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(self.right_frame, orient="vertical", 
                                command=self.stats_text.yview)
        scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.stats_text.configure(yscrollcommand=scrollbar.set)

    def get_price_trends(self, district, room_type):
        district_data = self.df[self.df['Bölge'] == district]
        avg_price_per_m2 = district_data['Fiyat'] / district_data['m² (Brüt)']
        return f"Average Price/m²: {avg_price_per_m2.mean():,.2f} TL"

    def get_room_distribution(self, district):
        district_data = self.df[self.df['Bölge'] == district]
        room_counts = district_data['Oda Sayısı'].value_counts()
        total = len(district_data)
        return "\nRoom Type Distribution:\n" + \
               "\n".join([f"{room}: {(count/total)*100:.1f}%" 
                         for room, count in room_counts.items()])

    def get_size_stats(self, district, room_type):
        similar = self.df[
            (self.df['Bölge'] == district) & 
            (self.df['Oda Sayısı'] == room_type)
        ]
        if len(similar) > 0:
            return f"\nSize Statistics for {room_type}:\n" + \
                   f"Min: {similar['m² (Brüt)'].min():.0f}m²\n" + \
                   f"Avg: {similar['m² (Brüt)'].mean():.0f}m²\n" + \
                   f"Max: {similar['m² (Brüt)'].max():.0f}m²"
        return ""

    def get_price_by_size(self, district, room_type):
        district_data = self.df[
            (self.df['Bölge'] == district) & 
            (self.df['Oda Sayısı'] == room_type)
        ]
        
        if len(district_data) > 0:
            small = district_data[district_data['m² (Brüt)'] <= district_data['m² (Brüt)'].quantile(0.33)]
            medium = district_data[(district_data['m² (Brüt)'] > district_data['m² (Brüt)'].quantile(0.33)) & 
                                 (district_data['m² (Brüt)'] <= district_data['m² (Brüt)'].quantile(0.66))]
            large = district_data[district_data['m² (Brüt)'] > district_data['m² (Brüt)'].quantile(0.66)]
            
            return f"\nPrice Ranges by Size:\n" + \
                   f"Small (≤{small['m² (Brüt)'].max():.0f}m²): {small['Fiyat'].mean():,.0f} TL\n" + \
                   f"Medium: {medium['Fiyat'].mean():,.0f} TL\n" + \
                   f"Large (≥{large['m² (Brüt)'].min():.0f}m²): {large['Fiyat'].mean():,.0f} TL"
        return ""

    def update_stats_display(self, range_text, price_trend, room_dist, size_stats, price_by_size):
        self.stats_text.delete('1.0', tk.END)
        self.stats_text.insert(tk.END, f"{range_text}\n\n")
        self.stats_text.insert(tk.END, f"{price_trend}\n")
        self.stats_text.insert(tk.END, f"{room_dist}\n")
        self.stats_text.insert(tk.END, f"{size_stats}\n")
        self.stats_text.insert(tk.END, f"{price_by_size}")

    def predict_price(self):
        """
        Main prediction function that:
        1. Gets user inputs
        2. Prepares features
        3. Makes prediction using selected model
        4. Shows results and relevant statistics
        """
        try:
            # Map selected model name to internal model key
            model_map = {
                'Linear Regression': 'linear',
                'Random Forest': 'random_forest',
                'Support Vector Regression': 'svr',
                'Gradient Boosting': 'gradient_boosting',
                'XGBoost': 'xgboost',
                'Neural Network': 'neural_network'
            }
            model_name = model_map[self.model_choice.get()]
            model, columns, scaler = self.models[model_name]
            
            # Get user inputs
            area = float(self.area.get())
            room_type = self.rooms.get()
            district = self.bolge.get()
            
            # Prepare feature vector
            features = pd.DataFrame(columns=columns, data=np.zeros((1, len(columns))))
            
            # Use loc[] accessor to set values
            features.loc[0, 'm² (Brüt)'] = area
            features.loc[0, f'oda_{room_type}'] = 1
            features.loc[0, f'bolge_{district}'] = 1
            
            # Scale features if needed
            if model_name in ['linear', 'svr']:
                features['m² (Brüt)'] = scaler.transform(features[['m² (Brüt)']])
            
            # Convert DataFrame to numpy array to avoid the warning
            features_array = features.to_numpy()
            
            # Make prediction
            prediction = model.predict(features_array)[0]
            
            # Find similar properties for comparison
            similar = self.df[
                (self.df['Bölge'] == district) & 
                (self.df['Oda Sayısı'] == room_type) & 
                (self.df['m² (Brüt)'] >= area * 0.8) & 
                (self.df['m² (Brüt)'] <= area * 1.2)
            ]
            
            # Calculate and display statistics
            if len(similar) > 0:
                min_price = similar['Fiyat'].min()
                max_price = similar['Fiyat'].max()
                avg_price = similar['Fiyat'].mean()
                count = len(similar)
                
                range_text = f"\nSimilar properties in {district}:\n" \
                           f"Count: {count}\n" \
                           f"Min: {min_price:,.2f} TL\n" \
                           f"Avg: {avg_price:,.2f} TL\n" \
                           f"Max: {max_price:,.2f} TL"
            else:
                range_text = f"\nNo similar properties found in {district}\n" \
                           f"for {room_type} with area around {area}m²"
            
            price_trend = self.get_price_trends(district, room_type)
            room_dist = self.get_room_distribution(district)
            size_stats = self.get_size_stats(district, room_type)
            price_by_size = self.get_price_by_size(district, room_type)
            
            # Update display
            self.result_label.config(
                text=f"Predicted Price ({self.model_choice.get()}): {prediction:,.2f} TL")
            self.update_stats_display(range_text, price_trend, room_dist, 
                                    size_stats, price_by_size)
            
        except Exception as e:
            self.result_label.config(text="Error: Please check your inputs")
            self.stats_text.delete('1.0', tk.END) 