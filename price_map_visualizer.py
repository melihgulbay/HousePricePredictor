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
        # Store the input DataFrame containing house data
        self.df = df
        
        # Dictionary to convert Turkish district names to English
        # This is needed because GeoJSON uses English names
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
            # Load geographic data for Istanbul districts
            gdf = gpd.read_file('istanbul_districts.geojson')
            
            # Convert district names to English for matching with house data
            gdf['name_eng'] = gdf['name'].map(self.turkish_to_english)
            
            # Calculate district-level statistics:
            # - Average price per district
            # - Number of houses per district
            district_stats = self.df.groupby('Bölge').agg({
                'Fiyat': ['mean', 'count']
            }).reset_index()
            district_stats.columns = ['Bölge', 'Fiyat', 'Count']
            
            # Merge price statistics with geographic data
            gdf = gdf.merge(district_stats, left_on='name_eng', right_on='Bölge', how='left')
            
            # Initialize the base map centered on Istanbul
            m = folium.Map(
                location=[41.0082, 28.9784],  # Istanbul coordinates
                zoom_start=10,
                tiles='CartoDB positron'  # Clean, light background
            )
            
            # Create the choropleth layer showing price distribution
            choropleth = folium.Choropleth(
                geo_data=json.loads(gdf.to_json()),
                name='choropleth',
                data=district_stats,
                columns=['Bölge', 'Fiyat'],
                key_on='feature.properties.name_eng',
                fill_color='YlOrRd',  # Yellow to Red color scheme
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name='Average Price (TL)',
                highlight=True
            ).add_to(m)
            
            # Define styling for district polygons
            # Normal state
            style_function = lambda x: {'fillColor': '#ffffff', 
                                      'color':'#000000', 
                                      'fillOpacity': 0.1, 
                                      'weight': 0.1}
            # Highlighted state (on hover)
            highlight_function = lambda x: {'fillColor': '#000000', 
                                          'color':'#000000', 
                                          'fillOpacity': 0.50, 
                                          'weight': 0.1}
            
            # Format prices with thousands separator for better readability
            gdf['Fiyat_Formatted'] = gdf['Fiyat'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A")
            
            # Create a feature group for interactive elements
            fg = folium.FeatureGroup(name="House Data")
            
            # Add interactive features for each district
            for idx, row in gdf.iterrows():
                # Get all houses in current district
                district_houses = self.df[self.df['Bölge'] == row['name_eng']]
                
                # Create HTML table showing individual house details
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
                
                # Add each house's details to the table
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
                
                # Add district polygon with:
                # - Hover tooltip showing summary statistics
                # - Clickable popup showing detailed house listings
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
            
            # Add the feature group to the map
            fg.add_to(m)
            
            # Save and display the map
            map_path = 'istanbul_price_map.html'
            m.save(map_path)
            webbrowser.open('file://' + os.path.realpath(map_path))
            
            return True, None
            
        except Exception as e:
            return False, str(e) 