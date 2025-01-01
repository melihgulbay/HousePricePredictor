import folium
from folium import plugins
import geopandas as gpd
import json
import webbrowser
import os
import tkinter as tk
import pandas as pd

class PriceMapVisualizer:
    def __init__(self, df):
        self.df = df
        self.turkish_to_english = {
            'Adalar': 'Adalar',
            'Arnavutköy': 'Arnavutkoy',
            'Ataşehir': 'Atasehir',
            'Avcılar': 'Avcilar',
            'Bağcılar': 'Bagcilar',
            'Bahçelievler': 'Bahcelievler',
            'Bakırköy': 'Bakirkoy',
            'Başakşehir': 'Basaksehir',
            'Bayrampaşa': 'Bayrampasa',
            'Beşiktaş': 'Besiktas',
            'Beykoz': 'Beykoz',
            'Beylikdüzü': 'Beylikduzu',
            'Beyoğlu': 'Beyoglu',
            'Büyükçekmece': 'Buyukcekmece',
            'Çatalca': 'Catalca',
            'Çekmeköy': 'Cekmekoy',
            'Esenler': 'Esenler',
            'Esenyurt': 'Esenyurt',
            'Eyüpsultan': 'Eyupsultan',
            'Fatih': 'Fatih',
            'Gaziosmanpaşa': 'Gaziosmanpasa',
            'Güngören': 'Gungoren',
            'Kadıköy': 'Kadikoy',
            'Kağıthane': 'Kagithane',
            'Kartal': 'Kartal',
            'Küçükçekmece': 'Kucukcekmece',
            'Maltepe': 'Maltepe',
            'Pendik': 'Pendik',
            'Sancaktepe': 'Sancaktepe',
            'Sarıyer': 'Sariyer',
            'Silivri': 'Silivri',
            'Sultanbeyli': 'Sultanbeyli',
            'Sultangazi': 'Sultangazi',
            'Şile': 'Sile',
            'Şişli': 'Sisli',
            'Tuzla': 'Tuzla',
            'Ümraniye': 'Umraniye',
            'Üsküdar': 'Uskudar',
            'Zeytinburnu': 'Zeytinburnu'
        }

    def create_price_map(self):
        try:
            # Read GeoJSON file
            gdf = gpd.read_file('istanbul_districts.geojson')
            
            # Convert GeoJSON district names to English
            gdf['name_eng'] = gdf['name'].map(self.turkish_to_english)
            
            # Calculate average prices and counts by district
            district_stats = self.df.groupby('Bölge').agg({
                'Fiyat': ['mean', 'count']
            }).reset_index()
            district_stats.columns = ['Bölge', 'Fiyat', 'Count']
            
            # Merge price data with geodataframe using English names
            gdf = gdf.merge(district_stats, left_on='name_eng', right_on='Bölge', how='left')
            
            # Create a Folium map centered on Istanbul
            m = folium.Map(
                location=[41.0082, 28.9784],
                zoom_start=10,
                tiles='CartoDB positron'
            )
            
            # Create choropleth layer
            choropleth = folium.Choropleth(
                geo_data=json.loads(gdf.to_json()),
                name='choropleth',
                data=district_stats,
                columns=['Bölge', 'Fiyat'],
                key_on='feature.properties.name_eng',
                fill_color='YlOrRd',
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name='Average Price (TL)',
                highlight=True
            ).add_to(m)
            
            # Add hover functionality
            style_function = lambda x: {'fillColor': '#ffffff', 
                                      'color':'#000000', 
                                      'fillOpacity': 0.1, 
                                      'weight': 0.1}
            highlight_function = lambda x: {'fillColor': '#000000', 
                                          'color':'#000000', 
                                          'fillOpacity': 0.50, 
                                          'weight': 0.1}
            
            # Format price with thousands separator
            gdf['Fiyat_Formatted'] = gdf['Fiyat'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A")
            
            # Add district names and stats on hover
            NIL = folium.features.GeoJson(
                json.loads(gdf.to_json()),
                style_function=style_function,
                control=False,
                highlight_function=highlight_function,
                tooltip=folium.features.GeoJsonTooltip(
                    fields=['name', 'name_eng', 'Fiyat_Formatted', 'Count'],
                    aliases=['District (TR):', 'District:', 'Average Price (TL):', 'Number of Houses:'],
                    style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
                )
            )
            m.add_child(NIL)
            m.keep_in_front(NIL)
            
            # Save the map
            map_path = 'istanbul_price_map.html'
            m.save(map_path)
            
            # Open in default web browser
            webbrowser.open('file://' + os.path.realpath(map_path))
            
            return True, None
            
        except Exception as e:
            return False, str(e) 