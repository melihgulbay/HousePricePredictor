import tkinter as tk
from tkinter import ttk
import pickle
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages
import tkinter.messagebox
from price_map_visualizer import PriceMapVisualizer
from gui_visualization import GUIVisualization
from gui_metrics import GUIMetrics
from gui_prediction import GUIPrediction

# Main GUI class that orchestrates all components
class HousePricePredictorGUI:
    def __init__(self, root):
        # Initialize main window settings
        self.root = root
        self.root.title("Istanbul House Price Predictor")
        self.root.geometry("1200x900")  # Set window size
        self.root.configure(bg='#f0f0f0')
        
        # Configure styles for consistent UI appearance
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
        
        # Create tabbed interface for organizing different features
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

        # Load trained models and their performance metrics
        with open('house_price_models.pkl', 'rb') as f:
            self.models_data = pickle.load(f)
        
        # Load and preprocess the dataset
        self.df = pd.read_csv('house_prices.csv', sep=';', encoding='utf-8')
        self.df['Fiyat'] = self.df['Fiyat'].str.replace(' TL', '').str.replace('.', '').str.replace(',', '.').astype(float)
        
        # Initialize prediction handler
        self.prediction_handler = GUIPrediction(prediction_frame, self.models_data, self.df)

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
        
        # Initialize metrics handler
        self.metrics_handler = GUIMetrics(self.models_data.get('metrics', {}), metrics_frame)
        
        # Store the current figure for PDF export
        self.current_figure = None

        # Initialize visualization handler
        self.viz_handler = GUIVisualization(self.df, self.plot_frame)

        # Configure grid weights for expansion
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(1, weight=1)

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
