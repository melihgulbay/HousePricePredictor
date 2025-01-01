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

class HousePricePredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Istanbul House Price Predictor")
        self.root.geometry("1200x900")  # Wider window
        self.root.configure(bg='#f0f0f0')
        
        # Configure styles
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
        
        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(container)
        self.notebook.grid(row=1, column=0, columnspan=2, sticky=(tk.N, tk.S, tk.E, tk.W))

        # Create main prediction tab
        prediction_frame = ttk.Frame(self.notebook)
        self.notebook.add(prediction_frame, text='Price Prediction')

        # Create visualization tab
        viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(viz_frame, text='Visualizations')

        # Create comparison tab
        comparison_frame = ttk.Frame(self.notebook)
        self.notebook.add(comparison_frame, text='District Comparison')
        
        # **Create Model Metrics Tab**
        metrics_frame = ttk.Frame(self.notebook)
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
        ttk.Button(viz_buttons_frame, text="District Comparison", 
                   command=self.show_district_comparison).pack(side=tk.LEFT, padx=5)
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
        
        # Load the models and metrics
        with open('house_price_models.pkl', 'rb') as f:
            self.models_data = pickle.load(f)
        self.models = {k: v for k, v in self.models_data.items() if k != 'metrics'}
        self.metrics = self.models_data.get('metrics', {})
        
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

        # Setup comparison tab
        self.setup_comparison_tab(comparison_frame)

        # **Setup Model Metrics Tab**
        self.setup_model_metrics_tab(metrics_frame)

        # Store the current figure for PDF export
        self.current_figure = None

    def setup_model_metrics_tab(self, frame):
        """
        Sets up the Model Metrics tab with a Treeview widget and performance visualization.
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
        try:
            # Update model selection
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
            
            # Scale features if using Linear Regression or SVR
            if model_name in ['linear', 'svr']:
                features['m² (Brüt)'] = scaler.transform(features[['m² (Brüt)']])
            
            # Convert DataFrame to numpy array to avoid the warning
            features_array = features.to_numpy()
            
            # Make prediction
            prediction = model.predict(features_array)[0]
            
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
            
            # Update display using the new method
            self.result_label.config(
                text=f"Predicted Price ({self.model_choice.get()}): {prediction:,.2f} TL")
            self.update_stats_display(range_text, price_trend, room_dist, 
                                    size_stats, price_by_size)
            
        except Exception as e:
            self.result_label.config(text="Error: Please check your inputs")
            self.stats_text.delete('1.0', tk.END)

    def create_plot_canvas(self, frame):
        # Clear existing plot frame
        for widget in frame.winfo_children():
            widget.destroy()
        
        # Create larger figure (increased from 10,6 to 12,8)
        fig = Figure(figsize=(12, 8), dpi=100)  # Added explicit DPI setting
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Store the figure reference
        self.current_figure = fig
        return fig

    def save_current_plot(self):
        if self.current_figure is None:
            return
            
        # Ask user for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            title="Save Plot as PDF"
        )
        
        if file_path:
            try:
                # Save the current figure as PDF
                with PdfPages(file_path) as pdf:
                    pdf.savefig(self.current_figure)
                    
                # Show success message
                tk.messagebox.showinfo(
                    "Success", 
                    "Plot successfully saved as PDF!"
                )
            except Exception as e:
                tk.messagebox.showerror(
                    "Error",
                    f"Failed to save PDF: {str(e)}"
                )

    def show_price_trends(self):
        fig = self.create_plot_canvas(self.plot_frame)
        ax = fig.add_subplot(111)
        
        # Calculate average prices by district
        avg_prices = self.df.groupby('Bölge')['Fiyat'].mean().sort_values(ascending=True)
        
        # Create line plot with larger markers and line width
        ax.plot(range(len(avg_prices)), avg_prices.values, marker='o', 
                markersize=8, linewidth=2)  # Increased marker and line size
        
        # Enhance text sizes
        ax.set_xticks(range(len(avg_prices)))
        ax.set_xticklabels(avg_prices.index, rotation=45, ha='right', fontsize=10)
        ax.set_title('Average House Prices by District', fontsize=14, pad=20)
        ax.set_ylabel('Price (TL)', fontsize=12)
        ax.tick_params(axis='y', labelsize=10)
        
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout with more padding
        fig.tight_layout(pad=2.0)

    def show_price_vs_area(self):
        fig = self.create_plot_canvas(self.plot_frame)
        ax = fig.add_subplot(111)
        
        # Create scatter plot with larger points and alpha
        sns.scatterplot(data=self.df, x='m² (Brüt)', y='Fiyat', ax=ax, 
                        alpha=0.6, s=100)  # Increased point size
        
        # Enhance text sizes
        ax.set_title('Price vs Area', fontsize=14, pad=20)
        ax.set_xlabel('Area (m²)', fontsize=12)
        ax.set_ylabel('Price (TL)', fontsize=12)
        ax.tick_params(axis='both', labelsize=10)
        
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        fig.tight_layout(pad=2.0)

    def show_district_heatmap(self):
        fig = self.create_plot_canvas(self.plot_frame)
        ax = fig.add_subplot(111)
        
        # Calculate average price per m² for each district
        price_per_m2 = self.df.groupby('Bölge')['Fiyat'].mean().reset_index()
        price_per_m2 = price_per_m2.pivot_table(index='Bölge', values='Fiyat')
        
        # Create heatmap with larger text
        sns.heatmap(price_per_m2, ax=ax, cmap='YlOrRd', 
                    annot=True, fmt='.0f', 
                    annot_kws={'size': 10})  # Increased annotation size
        
        # Enhance text sizes
        ax.set_title('Average Prices Heatmap by District', fontsize=14, pad=20)
        ax.tick_params(axis='both', labelsize=10)
        
        # Adjust layout
        fig.tight_layout(pad=2.0)

    def show_district_comparison(self):
        fig = self.create_plot_canvas(self.plot_frame)
        ax = fig.add_subplot(111)
        
        # Calculate average prices by district
        avg_prices = self.df.groupby('Bölge')['Fiyat'].mean().sort_values(ascending=True)
        
        # Create bar plot
        avg_prices.plot(kind='bar', ax=ax)
        ax.set_xticklabels(avg_prices.index, rotation=45, ha='right')
        ax.set_title('Average House Prices by District')
        ax.set_ylabel('Price (TL)')
        fig.tight_layout()

        # Store the comparison figure
        self.current_figure = fig

    def show_room_distribution(self):
        fig = self.create_plot_canvas(self.plot_frame)
        ax = fig.add_subplot(111)
        
        # Calculate room type distribution
        room_dist = self.df['Oda Sayısı'].value_counts()
        
        # Filter out very small segments (less than 1%)
        min_pct = 1.0
        other_pct = room_dist[room_dist/room_dist.sum()*100 < min_pct].sum()
        room_dist = room_dist[room_dist/room_dist.sum()*100 >= min_pct]
        
        # Add "Other" category if needed
        if other_pct > 0:
            room_dist['Other'] = other_pct
        
        # Sort values for better visualization
        room_dist = room_dist.sort_values(ascending=False)
        
        # Create pie chart with enhanced styling
        wedges, texts, autotexts = ax.pie(room_dist.values, 
                                         labels=room_dist.index,
                                         autopct='%1.1f%%',
                                         pctdistance=0.85,
                                         labeldistance=1.1,
                                         textprops={'fontsize': 10})  # Increased font size
        
        # Enhance text appearance
        plt.setp(autotexts, size=10, weight="bold")
        plt.setp(texts, size=10)
        
        # Add title with larger font
        ax.set_title('Distribution of Room Types', fontsize=14, pad=20)
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        
        # Adjust layout
        fig.tight_layout(pad=2.0)

    def setup_comparison_tab(self, frame):
        # Create left panel for district selection
        left_panel = ttk.Frame(frame, padding="10")
        left_panel.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W))

        # Create listbox with districts
        ttk.Label(left_panel, text="Select Districts to Compare:", 
                  style='Modern.TLabel').grid(row=0, column=0, pady=(0, 5))
        
        # Create listbox with scrollbar
        listbox_frame = ttk.Frame(left_panel)
        listbox_frame.grid(row=1, column=0)
        
        self.district_listbox = tk.Listbox(listbox_frame, selectmode=tk.MULTIPLE, 
                                          height=15, width=30)
        scrollbar = ttk.Scrollbar(listbox_frame, orient="vertical", 
                                 command=self.district_listbox.yview)
        
        self.district_listbox.config(yscrollcommand=scrollbar.set)
        self.district_listbox.pack(side=tk.LEFT, fill=tk.Y)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add districts to listbox
        for district in sorted(self.districts):
            self.district_listbox.insert(tk.END, district)

        # Create button frame
        button_frame = ttk.Frame(left_panel)
        button_frame.grid(row=2, column=0, pady=10)

        # Add comparison button
        ttk.Button(button_frame, text="Compare Districts", 
                   command=self.compare_districts, 
                   style='Modern.TButton').pack(side=tk.LEFT, padx=5)

        # Add download button
        ttk.Button(button_frame, text="Download as PDF", 
                   command=self.save_current_plot, 
                   style='Modern.TButton').pack(side=tk.LEFT, padx=5)

        # Create right panel for plots
        self.comparison_plot_frame = ttk.Frame(frame, padding="10")
        self.comparison_plot_frame.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))

        # Configure grid weights
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(0, weight=1)

    def compare_districts(self):
        # Get selected districts
        selections = self.district_listbox.curselection()
        if not selections:
            tk.messagebox.showwarning("No Selection", "Please select at least one district to compare.")
            return
        
        selected_districts = [self.district_listbox.get(i) for i in selections]
        
        # Create comparison plots
        fig = self.create_plot_canvas(self.comparison_plot_frame)
        
        # Create subplots (3 rows)
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 2, 1])
        ax1 = fig.add_subplot(gs[0])  # Room type distribution
        ax2 = fig.add_subplot(gs[1])  # Price distribution
        ax3 = fig.add_subplot(gs[2])  # Price per m²
        
        # Room type distribution comparison
        for district in selected_districts:
            district_data = self.df[self.df['Bölge'] == district]
            room_dist = district_data['Oda Sayısı'].value_counts()
            total = len(district_data)
            # Calculate percentages and keep only top 5 room types
            room_pcts = (room_dist / total * 100).nlargest(5)
            ax1.plot(room_pcts.index, room_pcts.values, marker='o', label=district)
        
        ax1.set_title('Room Type Distribution (Top 5)')
        ax1.set_ylabel('Percentage (%)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Price distribution comparison (using KDE plot)
        price_data = []
        for district in selected_districts:
            district_data = self.df[self.df['Bölge'] == district]
            # Normalize prices to millions for better readability
            prices_in_millions = district_data['Fiyat'] / 1_000_000
            sns.kdeplot(data=prices_in_millions, ax=ax2, label=district)
        
        ax2.set_title('Price Distribution')
        ax2.set_xlabel('Price (Million TL)')
        ax2.set_ylabel('Density')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Average price per m²
        avg_prices = []
        for district in selected_districts:
            district_df = self.df[self.df['Bölge'] == district]
            avg_price_per_m2 = (district_df['Fiyat'] / district_df['m² (Brüt)']).mean()
            avg_prices.append(avg_price_per_m2)
        
        # Create bar plot
        bars = ax3.bar(range(len(selected_districts)), avg_prices)
        ax3.set_xticks(range(len(selected_districts)))
        ax3.set_xticklabels(selected_districts, rotation=45)
        ax3.set_title('Average Price per m²')
        ax3.set_ylabel('TL/m²')
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # Add value labels on the bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom')
        
        # Adjust layout
        fig.tight_layout()

    def show_price_box_plot(self):
        """Shows price distribution across districts using box plots"""
        fig = self.create_plot_canvas(self.plot_frame)
        ax = fig.add_subplot(111)
        
        # Create box plot
        sns.boxplot(data=self.df, x='Bölge', y='Fiyat', ax=ax)
        
        # Customize the plot
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title('House Price Distribution by District', fontsize=14, pad=20)
        ax.set_xlabel('District', fontsize=12)
        ax.set_ylabel('Price (TL)', fontsize=12)
        
        # Format y-axis labels to show millions
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
        
        fig.tight_layout(pad=2.0)

    def show_area_distribution(self):
        """Shows area distribution using histogram and KDE"""
        fig = self.create_plot_canvas(self.plot_frame)
        ax = fig.add_subplot(111)
        
        # Create histogram with KDE
        sns.histplot(data=self.df, x='m² (Brüt)', kde=True, ax=ax)
        
        # Customize the plot
        ax.set_title('Distribution of House Areas', fontsize=14, pad=20)
        ax.set_xlabel('Area (m²)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        
        # Add mean and median lines
        mean_area = self.df['m² (Brüt)'].mean()
        median_area = self.df['m² (Brüt)'].median()
        
        ax.axvline(mean_area, color='red', linestyle='--', label=f'Mean: {mean_area:.0f}m²')
        ax.axvline(median_area, color='green', linestyle='--', label=f'Median: {median_area:.0f}m²')
        ax.legend(fontsize=10)
        
        fig.tight_layout(pad=2.0)

    def show_room_price_analysis(self):
        """Shows average price by room type with error bars"""
        fig = self.create_plot_canvas(self.plot_frame)
        ax = fig.add_subplot(111)
        
        # Calculate mean and standard error for each room type
        room_stats = self.df.groupby('Oda Sayısı')['Fiyat'].agg(['mean', 'count', 'std']).reset_index()
        room_stats['se'] = room_stats['std'] / np.sqrt(room_stats['count'])
        
        # Sort by mean price
        room_stats = room_stats.sort_values('mean', ascending=True)
        
        # Create bar plot with error bars
        bars = ax.bar(range(len(room_stats)), room_stats['mean'])
        ax.errorbar(range(len(room_stats)), room_stats['se'], 
                   fmt='none', color='black', capsize=5)
        
        # Customize the plot
        ax.set_xticks(range(len(room_stats)))
        ax.set_xticklabels(room_stats['Oda Sayısı'], rotation=45, ha='right')
        ax.set_title('Average Price by Room Type', fontsize=14, pad=20)
        ax.set_xlabel('Room Type', fontsize=12)
        ax.set_ylabel('Average Price (TL)', fontsize=12)
        
        # Format y-axis labels to show millions
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height/1e6:.1f}M',
                   ha='center', va='bottom')
        
        fig.tight_layout(pad=2.0)

    def show_price_map(self):
        visualizer = PriceMapVisualizer(self.df)
        success, error = visualizer.create_price_map()
        
        if not success:
            tk.messagebox.showerror("Error", f"Failed to create map: {error}")

    def show_price_per_m2(self):
        """Shows average price per square meter by district"""
        fig = self.create_plot_canvas(self.plot_frame)
        ax = fig.add_subplot(111)
        
        # Calculate price per m² for each property
        self.df['Price_per_m2'] = self.df['Fiyat'] / self.df['m² (Brüt)']
        
        # Calculate average price per m² for each district
        district_avg = self.df.groupby('Bölge')['Price_per_m2'].mean().sort_values(ascending=True)
        
        # Create bar plot
        bars = ax.bar(range(len(district_avg)), district_avg)
        
        # Customize the plot
        ax.set_xticks(range(len(district_avg)))
        ax.set_xticklabels(district_avg.index, rotation=45, ha='right')
        ax.set_title('Average Price per m² by District', fontsize=14, pad=20)
        ax.set_xlabel('District', fontsize=12)
        ax.set_ylabel('Price per m² (TL)', fontsize=12)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:,.0f}',
                   ha='center', va='bottom', rotation=0)
        
        fig.tight_layout(pad=2.0)

    def show_room_size_analysis(self):
        """Shows average size by room type"""
        fig = self.create_plot_canvas(self.plot_frame)
        ax = fig.add_subplot(111)
        
        # Calculate average size for each room type
        room_sizes = self.df.groupby('Oda Sayısı')['m² (Brüt)'].agg(['mean', 'std']).reset_index()
        room_sizes = room_sizes.sort_values('mean', ascending=True)
        
        # Create bar plot with error bars
        bars = ax.bar(range(len(room_sizes)), room_sizes['mean'])
        ax.errorbar(range(len(room_sizes)), room_sizes['mean'], 
                   yerr=room_sizes['std'], fmt='none', color='black', capsize=5)
        
        # Customize the plot
        ax.set_xticks(range(len(room_sizes)))
        ax.set_xticklabels(room_sizes['Oda Sayısı'], rotation=45, ha='right')
        ax.set_title('Average Size by Room Type', fontsize=14, pad=20)
        ax.set_xlabel('Room Type', fontsize=12)
        ax.set_ylabel('Average Size (m²)', fontsize=12)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}m²',
                   ha='center', va='bottom')
        
        fig.tight_layout(pad=2.0)

    def show_district_size_distribution(self):
        """Shows average size distribution across districts using bar plot with error bars"""
        fig = self.create_plot_canvas(self.plot_frame)
        ax = fig.add_subplot(111)
        
        # Calculate mean and standard deviation for each district
        district_stats = self.df.groupby('Bölge')['m² (Brüt)'].agg(['mean', 'std']).reset_index()
        district_stats = district_stats.sort_values('mean', ascending=True)  # Sort by average size
        
        # Create bar plot with error bars
        bars = ax.bar(range(len(district_stats)), district_stats['mean'])
        ax.errorbar(range(len(district_stats)), district_stats['mean'], 
                   yerr=district_stats['std'], fmt='none', color='black', capsize=5)
        
        # Customize the plot
        ax.set_xticks(range(len(district_stats)))
        ax.set_xticklabels(district_stats['Bölge'], rotation=45, ha='right')
        ax.set_title('Average House Size by District', fontsize=14, pad=20)
        ax.set_xlabel('District', fontsize=12)
        ax.set_ylabel('Average Size (m²)', fontsize=12)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}m²',
                   ha='center', va='bottom')
        
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        fig.tight_layout(pad=2.0)
