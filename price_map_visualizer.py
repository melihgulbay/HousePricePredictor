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
            
            # Create a feature group for the popups
            fg = folium.FeatureGroup(name="House Data")
            
            # Add district names, stats on hover, and house data on click
            for idx, row in gdf.iterrows():
                # Get house data for this district
                district_houses = self.df[self.df['Bölge'] == row['name_eng']]
                
                # Create HTML table for house data
                house_data_html = """
                <div style="max-height: 300px; overflow-y: auto;">
                    <h4>Houses in {}</h4>
                    <table style="width:100%; border-collapse: collapse;">
                        <tr>
                            <th style="border: 1px solid #ddd; padding: 8px;">Area (m²)</th>
                            <th style="border: 1px solid #ddd; padding: 8px;">Rooms</th>
                            <th style="border: 1px solid #ddd; padding: 8px;">Price (TL)</th>
                        </tr>
                """.format(row['name'])
                
                for _, house in district_houses.iterrows():
                    house_data_html += """
                        <tr>
                            <td style="border: 1px solid #ddd; padding: 8px;">{}</td>
                            <td style="border: 1px solid #ddd; padding: 8px;">{}</td>
                            <td style="border: 1px solid #ddd; padding: 8px;">{:,.0f}</td>
                        </tr>
                    """.format(
                        house['m² (Brüt)'],
                        house['Oda Sayısı'],
                        house['Fiyat']
                    )
                
                house_data_html += "</table></div>"
                
                # Create GeoJson with both tooltip and popup
                folium.GeoJson(
                    row.geometry.__geo_interface__,
                    style_function=style_function,
                    highlight_function=highlight_function,
                    tooltip=folium.Tooltip(
                        f"""
                        District (TR): {row['name']}<br>
                        District: {row['name_eng']}<br>
                        Average Price (TL): {row['Fiyat_Formatted']}<br>
                        Number of Houses: {row['Count']}
                        """,
                        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
                    ),
                    popup=folium.Popup(house_data_html, max_width=500)
                ).add_to(fg)
            
            fg.add_to(m)
            
            # Save the map
            map_path = 'istanbul_price_map.html'
            m.save(map_path)
            
            # Open in default web browser
            webbrowser.open('file://' + os.path.realpath(map_path))
            
            return True, None
            
        except Exception as e:
            return False, str(e) 