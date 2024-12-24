import tkinter as tk
from tkinter import ttk
import pickle
import pandas as pd
import numpy as np

class HousePricePredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("House Price Predictor")
        
        # Load the models
        with open('house_price_models.pkl', 'rb') as f:
            self.models = pickle.load(f)
        
        # Districts list with English characters
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
        
        # Room types list
        self.room_types = [
            'Stüdyo (1+0)', '1+1', '1.5+1', '2+0', '2+1', '2.5+1', '2+2',
            '3+0', '3+1', '3.5+1', '3+2', '3+3', '4+0', '4+1', '4.5+1',
            '4.5+2', '4+2', '4+3', '4+4', '5+1', '5.5+1', '5+2', '5+3',
            '5+4', '6+1', '6+2', '6.5+1', '6+3', '6+4', '7+1', '7+2',
            '7+3', '8+1', '8+2', '8+3', '8+4', '9+1', '9+2', '9+3',
            '9+4', '9+5', '9+6', '10+1', '10+2'
        ]
        
        # Load the dataset for statistics
        self.df = pd.read_csv('house_prices.csv', sep=';', encoding='utf-8')
        self.df['Fiyat'] = self.df['Fiyat'].str.replace(' TL', '').str.replace('.', '').str.replace(',', '.').astype(float)
        
        # Create GUI elements
        ttk.Label(root, text="Model:").grid(row=0, column=0, padx=5, pady=5)
        self.model_choice = ttk.Combobox(root, values=['Linear Regression', 'Random Forest'])
        self.model_choice.grid(row=0, column=1, padx=5, pady=5)
        self.model_choice.set('Random Forest')  # default value
        
        ttk.Label(root, text="Bölge:").grid(row=1, column=0, padx=5, pady=5)
        self.bolge = ttk.Combobox(root, values=self.districts)
        self.bolge.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(root, text="m² (Brüt):").grid(row=2, column=0, padx=5, pady=5)
        self.area = ttk.Entry(root)
        self.area.grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(root, text="Oda Sayısı:").grid(row=3, column=0, padx=5, pady=5)
        self.rooms = ttk.Combobox(root, values=self.room_types)
        self.rooms.grid(row=3, column=1, padx=5, pady=5)
        
        ttk.Button(root, text="Predict Price", command=self.predict_price).grid(row=4, column=0, columnspan=2, pady=10)
        
        # Create labels for different statistics
        self.result_label = ttk.Label(root, text="")
        self.result_label.grid(row=5, column=0, columnspan=2, pady=5)
        
        self.range_label = ttk.Label(root, text="", justify=tk.LEFT)
        self.range_label.grid(row=6, column=0, columnspan=2, pady=5)
        
        self.stats_label = ttk.Label(root, text="", justify=tk.LEFT)
        self.stats_label.grid(row=7, column=0, columnspan=2, pady=5)

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

    def predict_price(self):
        try:
            # Get selected model
            model_name = 'linear' if self.model_choice.get() == 'Linear Regression' else 'random_forest'
            model, columns, scaler = self.models[model_name]
            
            # Get input values
            area = float(self.area.get())
            room_type = self.rooms.get()
            district = self.bolge.get()
            
            # Create feature vector
            features = pd.DataFrame(columns=columns, data=np.zeros((1, len(columns))))
            
            # Use loc[] accessor to set values
            features.loc[0, 'm² (Brüt)'] = area
            features.loc[0, f'oda_{room_type}'] = 1
            features.loc[0, f'bolge_{district}'] = 1
            
            # Scale features if using Linear Regression
            if model_name == 'linear':
                features['m² (Brüt)'] = scaler.transform(features[['m² (Brüt)']])
            
            # Make prediction
            prediction = model.predict(features)[0]
            
            # Find similar properties
            similar = self.df[
                (self.df['Bölge'] == district) & 
                (self.df['Oda Sayısı'] == room_type) & 
                (self.df['m² (Brüt)'] >= area * 0.8) & 
                (self.df['m² (Brüt)'] <= area * 1.2)
            ]
            
            # Get basic range statistics
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
            
            # Get additional statistics
            price_trend = self.get_price_trends(district, room_type)
            room_dist = self.get_room_distribution(district)
            size_stats = self.get_size_stats(district, room_type)
            price_by_size = self.get_price_by_size(district, room_type)
            
            # Display all results
            self.result_label.config(text=f"Predicted Price ({self.model_choice.get()}): {prediction:,.2f} TL")
            self.range_label.config(text=range_text)
            self.stats_label.config(text=f"\n{price_trend}{room_dist}{size_stats}{price_by_size}")
            
        except Exception as e:
            self.result_label.config(text=f"Error: Please check your inputs")
            self.range_label.config(text="")
            self.stats_label.config(text="")
