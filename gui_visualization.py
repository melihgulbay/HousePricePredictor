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


class GUIVisualization:
    def __init__(self, df, plot_frame):
        self.df = df
        self.plot_frame = plot_frame
        self.current_figure = None

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
        """Creates line plot showing average prices by district"""
        fig = self.create_plot_canvas(self.plot_frame)
        ax = fig.add_subplot(111)
        
        # Calculate average prices by district
        avg_prices = self.df.groupby('Bölge')['Fiyat'].mean().sort_values(ascending=True)
        
        # Create line plot with larger markers and line width
        ax.plot(range(len(avg_prices)), avg_prices.values, marker='o', 
                markersize=8, linewidth=2)
        
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
        """Creates scatter plot of price vs area"""
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
        """Creates heatmap showing price distribution across districts"""
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

    def show_room_distribution(self):
        """Creates pie chart showing room type distribution"""
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
        
        # Create pie chart with enhanced styling
        wedges, texts, autotexts = ax.pie(room_dist.values, 
                                         labels=room_dist.index,
                                         autopct='%1.1f%%',
                                         pctdistance=0.85,
                                         labeldistance=1.1)
        
        # Enhance text appearance
        plt.setp(autotexts, size=10, weight="bold")
        plt.setp(texts, size=10)
        
        # Add title with larger font
        ax.set_title('Distribution of Room Types', fontsize=14, pad=20)
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        
        # Adjust layout
        fig.tight_layout(pad=2.0)

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
        ax.errorbar(range(len(room_stats)), room_stats['mean'], 
                   yerr=room_stats['se'], fmt='none', color='black', capsize=5)
        
        # Customize the plot
        ax.set_xticks(range(len(room_stats)))
        ax.set_xticklabels(room_stats['Oda Sayısı'], rotation=45, ha='right')
        ax.set_title('Average Price by Room Type', fontsize=14, pad=20)
        ax.set_xlabel('Room Type', fontsize=12)
        ax.set_ylabel('Average Price (TL)', fontsize=12)
        
        # Format y-axis labels to show millions
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
        
        fig.tight_layout(pad=2.0)

    def show_price_map(self):
        """Shows price distribution on a map using PriceMapVisualizer"""
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
        """Shows average size distribution across districts"""
        fig = self.create_plot_canvas(self.plot_frame)
        ax = fig.add_subplot(111)
        
        # Calculate mean and standard deviation for each district
        district_stats = self.df.groupby('Bölge')['m² (Brüt)'].agg(['mean', 'std']).reset_index()
        district_stats = district_stats.sort_values('mean', ascending=True)
        
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