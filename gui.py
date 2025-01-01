import tkinter as tk
from tkinter import ttk
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from tkinter import filedialog
import tkinter.messagebox
import geopandas as gpd
import contextily as ctx
import folium
from folium import plugins
import json
import webbrowser
import os
from price_map_visualizer import PriceMapVisualizer
from gui_visualization import GUIVisualization

class HousePricePredictorGUI:
    def __init__(self, root):
        # Initialize main window settings
        self.root = root
        self.root.title("Istanbul House Price Predictor")
        self.root.geometry("1200x900")  # Set window size
        self.root.configure(bg='#f0f0f0')
        
        # Configure visual styles for widgets
        style = ttk.Style()
        style.configure('Modern.TLabel', font=('Helvetica', 12), padding=5)
        style.configure('Header.TLabel', font=('Helvetica', 16, 'bold'), padding=10)
        style.configure('Result.TLabel', font=('Helvetica', 12), padding=10, background='#ffffff')
        style.configure('Modern.TButton', font=('Helvetica', 12, 'bold'), padding=10)
        style.configure('Modern.TCombobox', font=('Helvetica', 11))
        
        # Create main container that will hold left and right frames
        container = ttk.Frame(root, padding="20")
        container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)
        
        # Title spanning both columns
        title_label = ttk.Label(container, text="Istanbul House Price Predictor", 
                               style='Header.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=20)
        
        # Create tabbed interface for different features
        self.notebook = ttk.Notebook(container)
        self.notebook.grid(row=1, column=0, columnspan=2, sticky=(tk.N, tk.S, tk.E, tk.W))

        # Initialize different tabs for various functionalities
        prediction_frame = ttk.Frame(self.notebook)  # Main prediction interface
        viz_frame = ttk.Frame(self.notebook)        # Data visualization
        metrics_frame = ttk.Frame(self.notebook)    # Model performance metrics

        # Add tabs to notebook
        self.notebook.add(prediction_frame, text='Price Prediction')
        self.notebook.add(viz_frame, text='Visualizations')
        self.notebook.add(metrics_frame, text='Model Metrics')

        # Initialize existing frames
        # Move existing left_frame and right_frame into prediction_frame
        left_frame = ttk.LabelFrame(prediction_frame, text="Input Parameters", padding="20")
        left_frame.grid(row=0, column=0, sticky=(tk.N, tk.W, tk.E, tk.S), padx=10)
        
        right_frame = ttk.LabelFrame(prediction_frame, text="Results", padding="20")
        right_frame.grid(row=0, column=1, sticky=(tk.N, tk.W, tk.E, tk.S), padx=10)

        # Add visualization buttons
        viz_buttons_frame = ttk.Frame(viz_frame)
        viz_buttons_frame.grid(row=0, column=0, pady=10)

        ttk.Button(viz_buttons_frame, text="Price Trends", 
                   command=self.show_price_trends).pack(side=tk.LEFT, padx=5)
        ttk.Button(viz_buttons_frame, text="Price vs Area", 
                   command=self.show_price_vs_area).pack(side=tk.LEFT, padx=5)
        ttk.Button(viz_buttons_frame, text="District Heatmap", 
                   command=self.show_district_heatmap).pack(side=tk.LEFT, padx=5)
        ttk.Button(viz_buttons_frame, text="Room Distribution", 
                   command=self.show_room_distribution).pack(side=tk.LEFT, padx=5)
        ttk.Button(viz_buttons_frame, text="Price Map", 
                   command=self.show_price_map).pack(side=tk.LEFT, padx=5)
        ttk.Button(viz_buttons_frame, text="Price Box Plot", 
                   command=self.show_price_box_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(viz_buttons_frame, text="Area Distribution", 
                   command=self.show_area_distribution).pack(side=tk.LEFT, padx=5)
        ttk.Button(viz_buttons_frame, text="Room Price Analysis", 
                   command=self.show_room_price_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(viz_buttons_frame, text="Price per m²", 
                   command=self.show_price_per_m2).pack(side=tk.LEFT, padx=5)
        ttk.Button(viz_buttons_frame, text="Room Size Analysis", 
                   command=self.show_room_size_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(viz_buttons_frame, text="District Size Distribution", 
                   command=self.show_district_size_distribution).pack(side=tk.LEFT, padx=5)

        # Add download button next to visualization buttons
        ttk.Button(viz_buttons_frame, text="Download as PDF", 
                   command=self.save_current_plot).pack(side=tk.LEFT, padx=5)

        # Frame for the plots
        self.plot_frame = ttk.Frame(viz_frame)
        self.plot_frame.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # Load trained models and their performance metrics
        with open('house_price_models.pkl', 'rb') as f:
            self.models_data = pickle.load(f)
        self.models = {k: v for k, v in self.models_data.items() if k != 'metrics'}
        self.metrics = self.models_data.get('metrics', {})
        
        # Define available districts and room types
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
        
        # Load and preprocess the dataset
        self.df = pd.read_csv('house_prices.csv', sep=';', encoding='utf-8')
        self.df['Fiyat'] = self.df['Fiyat'].str.replace(' TL', '').str.replace('.', '').str.replace(',', '.').astype(float)
        
        # Input elements in left frame
        ttk.Label(left_frame, text="Model:", style='Modern.TLabel').grid(row=0, column=0, padx=15, pady=10, sticky=tk.W)
        self.model_choice = ttk.Combobox(
            left_frame, 
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
        
        ttk.Label(left_frame, text="District:", style='Modern.TLabel').grid(row=1, column=0, padx=15, pady=10, sticky=tk.W)
        self.bolge = ttk.Combobox(left_frame, values=self.districts, width=30, style='Modern.TCombobox')
        self.bolge.grid(row=1, column=1, padx=15, pady=10)
        
        ttk.Label(left_frame, text="Area (m²):", style='Modern.TLabel').grid(row=2, column=0, padx=15, pady=10, sticky=tk.W)
        self.area = ttk.Entry(left_frame, width=32, font=('Helvetica', 11))
        self.area.grid(row=2, column=1, padx=15, pady=10)
        
        ttk.Label(left_frame, text="Room Type:", style='Modern.TLabel').grid(row=3, column=0, padx=15, pady=10, sticky=tk.W)
        self.rooms = ttk.Combobox(left_frame, values=self.room_types, width=30, style='Modern.TCombobox')
        self.rooms.grid(row=3, column=1, padx=15, pady=10)
        
        # Predict Button at bottom of left frame
        predict_button = ttk.Button(left_frame, text="Predict Price", 
                                  command=self.predict_price, style='Modern.TButton')
        predict_button.grid(row=4, column=0, columnspan=2, pady=20)
        
        # Results elements in right frame
        self.result_label = ttk.Label(right_frame, style='Result.TLabel', 
                                    background='#ffffff', relief='solid')
        self.result_label.grid(row=0, column=0, pady=10, sticky=(tk.W, tk.E))
        
        # Create Text widget for statistics in right frame
        self.stats_text = tk.Text(right_frame, wrap=tk.WORD, height=30,
                                font=('Helvetica', 11), relief='solid')
        self.stats_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add scrollbar to right frame
        scrollbar = ttk.Scrollbar(right_frame, orient="vertical", 
                                command=self.stats_text.yview)
        scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.stats_text.configure(yscrollcommand=scrollbar.set)
        
        # Configure grid weights for expansion
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(1, weight=1)
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)

        # Setup Model Metrics Tab
        self.setup_model_metrics_tab(metrics_frame)

        # Store the current figure for PDF export
        self.current_figure = None

        # Initialize visualization handler
        self.viz_handler = GUIVisualization(self.df, self.plot_frame)

    def setup_model_metrics_tab(self, frame):
        """
        Creates the Model Metrics tab showing:
        1. Performance metrics table
        2. Comparative visualization of model performance
        """
        # Create container frames
        table_frame = ttk.Frame(frame)
        table_frame.pack(fill=tk.X, padx=10, pady=5)
        
        graph_frame = ttk.Frame(frame)
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create a Treeview widget
        columns = ('Model', 'CV Score Mean', 'CV Score Std', 'RMSE', 'MAE', 'R2', 'Prediction Std')
        tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=6)
        
        # Define headings
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150, anchor='center')
        
        # Add vertical scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(fill=tk.X, expand=True)

        # Insert metrics data into the Treeview with formatted numbers
        for model_name, metric in self.metrics.items():
            tree.insert('', tk.END, values=(
                model_name.replace('_', ' ').title(),
                f"{metric['cv_score_mean']:.3f}",
                f"{metric['cv_score_std']:.3f}",
                f"{metric['rmse']:,.2f}",
                f"{metric['mae']:,.2f}",
                f"{metric['r2']:.3f}",
                f"{metric['prediction_std']:,.2f}" if metric['prediction_std'] is not None else 'N/A'
            ))

        # Configure tag for alternating row colors
        tree.tag_configure('oddrow', background='#f0f0f0')
        tree.tag_configure('evenrow', background='#ffffff')
        
        # Apply alternating row colors
        for i, item in enumerate(tree.get_children()):
            tree.item(item, tags=('evenrow' if i % 2 == 0 else 'oddrow',))

        # Create the comparison graph
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Prepare data for plotting
        models = []
        cv_scores = []
        rmse_scores = []
        r2_scores = []
        
        for model_name, metric in self.metrics.items():
            models.append(model_name.replace('_', ' ').title())
            cv_scores.append(metric['cv_score_mean'])
            rmse_scores.append(metric['rmse'])
            r2_scores.append(metric['r2'])
        
        # Set positions for bars
        x = np.arange(len(models))
        width = 0.25
        
        # Create grouped bar chart
        ax.bar(x - width, cv_scores, width, label='CV Score', color='skyblue')
        ax.bar(x, r2_scores, width, label='R² Score', color='lightgreen')
        ax.bar(x + width, rmse_scores / max(rmse_scores), width, 
               label='RMSE (normalized)', color='salmon')
        
        # Customize the plot
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add value labels on the bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8)
        
        # Add value labels for each set of bars
        for container in ax.containers:
            add_value_labels(container)
        
        # Adjust layout
        fig.tight_layout()
        
        # Create canvas and add to frame
        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Store the figure reference
        self.current_figure = fig

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

    def show_price_trends(self):
        self.viz_handler.show_price_trends()

    def show_price_vs_area(self):
        self.viz_handler.show_price_vs_area()

    def show_district_heatmap(self):
        self.viz_handler.show_district_heatmap()

    def show_room_distribution(self):
        self.viz_handler.show_room_distribution()

    def show_price_box_plot(self):
        self.viz_handler.show_price_box_plot()

    def show_area_distribution(self):
        self.viz_handler.show_area_distribution()

    def show_room_price_analysis(self):
        self.viz_handler.show_room_price_analysis()

    def show_price_map(self):
        self.viz_handler.show_price_map()

    def show_price_per_m2(self):
        self.viz_handler.show_price_per_m2()

    def show_room_size_analysis(self):
        self.viz_handler.show_room_size_analysis()

    def show_district_size_distribution(self):
        self.viz_handler.show_district_size_distribution()

    def save_current_plot(self):
        self.viz_handler.save_current_plot()
