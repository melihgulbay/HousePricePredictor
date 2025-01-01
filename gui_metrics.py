import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class GUIMetrics:
    def __init__(self, metrics, frame):
        self.metrics = metrics
        self.setup_model_metrics_tab(frame)
        self.current_figure = None

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
        self.create_comparison_graph(graph_frame)

    def create_comparison_graph(self, frame):
        """Creates and displays the model comparison graph"""
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
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Store the figure reference
        self.current_figure = fig 