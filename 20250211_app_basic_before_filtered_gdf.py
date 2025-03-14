from flask import Flask, render_template, request, Markup
import geopandas as gpd
import folium
import random
import pandas as pd
from sqlalchemy import create_engine
import psycopg2
from shapely.wkt import loads  # Required for geometry conversion
import branca

app = Flask(__name__)

# Database connection details
sign_in = {
    "database": "postgres",
    "user": "postgres",
    "password": "1234567h",
    "host": "localhost",
    "port": "5434"
}

# Create database engine
signin_info = f"postgresql+psycopg2://{sign_in['user']}:{sign_in['password']}@{sign_in['host']}:{sign_in['port']}/{sign_in['database']}"
engine1 = create_engine(signin_info)

def retrieve_streak(streak_id, engine, view_name):
    """Retrieve streak data from the database."""
    query = f'''SELECT * FROM public.{view_name} WHERE "Streak_ID" = {streak_id}'''
    result = engine.execute(query)
    df_result = pd.DataFrame(result.fetchall(), columns=result.keys())

    if df_result.empty:
        return None  # Return None if no data is found

    return df_result

def convert_sql_to_gdf(df, geom_column):
    """Convert a SQL DataFrame to a GeoDataFrame using the given geometry column."""
    df['geometry'] = df[geom_column].str.replace(',', ' ').apply(loads)  
    gdf = gpd.GeoDataFrame(df, geometry=df['geometry'], crs="EPSG:4326")
    return gdf

def find_optimum_stadiums(gdf):
    
    categories = gdf['date'].unique()

    gdf = gdf[gdf["geometry"]!= None]

    groups = {cat: gdf[gdf['date'] == cat] for cat in categories}

    names = {cat: group['unique_id'].tolist() for cat, group in groups.items()}

    # Extract coordinates for distance calculations
    coords = {cat: group.geometry.apply(lambda x: (x.x, x.y)).tolist() for cat, group in groups.items()}

    #print(coords,date_categories)
    
    # Function to find the shortest route
    
    route = []
    total_distance = 0

    # Start at the red category
    current_coords = np.array(coords[categories[0]])
    current_names = names[categories[0]]

    for i in range(len(categories) - 1):
        next_category = categories[i + 1]
        next_coords = np.array(coords[next_category])
        next_names = names[next_category]

        # Calculate pairwise distances
        distances = cdist(current_coords, next_coords)
        min_dist_idx = np.unravel_index(distances.argmin(), distances.shape)

        # Append the chosen point's name to the route and update total distance
        route.append((categories[i], current_names[min_dist_idx[0]]))
        total_distance += distances[min_dist_idx]

        # Update current_coords and current_names to the selected next point
        current_coords = next_coords[[min_dist_idx[1]], :]
        current_names = [next_names[min_dist_idx[1]]]

        # Add the final point
        route.append((categories[-1], current_names[0]))

        

    # Calculate the shortest route
    

    names_list = [name for _, name in route]
    
    #print("Route (category and name):", route)
    print("Total Distance:", total_distance)
    print(names_list)
    
    return route,total_distance,names_list




def plot_streak_map(gdf, jitter_amount=0.005):
    """Generate a Folium map and return it as an embedded HTML string."""
    if gdf is None or gdf.empty:
        return None  # No data to plot

    gdf = gdf.to_crs(epsg=4326)
    gdf['date'] = gdf['date'].astype(str)

    unique_dates = gdf['date'].unique()
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'black', 'yellow', 'brown', 'olive', 'cyan', 'mediumseagreen', 'white']
    color_map = {date: colors[i % len(colors)] for i, date in enumerate(unique_dates)}

    # Create map centered on the data
    m = folium.Map(location=[gdf.geometry.y.mean(), gdf.geometry.x.mean()], zoom_start=8)

    for _, row in gdf.iterrows():
        if row["geometry"] is not None:
            jittered_lat = row.geometry.y + random.uniform(-jitter_amount, jitter_amount)
            jittered_lon = row.geometry.x + random.uniform(-jitter_amount, jitter_amount)

            folium.CircleMarker(
                location=[jittered_lat, jittered_lon],
                radius=6,
                color='black',
                fill=True,
                fillColor=color_map.get(row['date'], 'gray'),
                fillOpacity=0.9,
                popup=folium.Popup(f"Date: {row['date']}<br>Home: {row['home']} vs Away: {row['away']}")
            ).add_to(m)
    
        # Create a custom legend as HTML
    legend_html = """
    <div style="
        position: fixed;
        bottom: 50px;
        left: 50px;
        width: 250px;
        height: auto;
        background-color: white;
        border: 2px solid black;
        border-radius: 8px;
        padding: 10px;
        font-size: 14px;
        z-index: 1000;
    ">
        <h4 style="margin: 0; text-align: center;">Legend</h4>
        <ul style="list-style-type: none; padding: 0; margin: 0;">
    """
    for date, color in color_map.items():
        legend_html += f"""
        <li style="margin: 5px 0;">
            <span style="display: inline-block; width: 20px; height: 20px; background-color: {color}; margin-right: 10px; border: 1px solid black;"></span>
            {date}
        </li>
        """
    legend_html += "</ul></div>"

    # Add the legend to the map
    legend = branca.element.MacroElement()
    legend._template = branca.element.Template(legend_html)
    m.get_root().html.add_child(folium.Element(legend_html))
    
    
    return m._repr_html_()

@app.route("/", methods=["GET", "POST"])
def index():
    combined_streak_view = "combined_stadium_streaks_view2"
    map_html = None
    error_message = None

    if request.method == "POST":
        streak_id = request.form.get("streak_id")

        if streak_id:
            try:
                streak_id = int(streak_id)  # Convert input to integer
                selected_streak = retrieve_streak(streak_id, engine1, combined_streak_view)
                
                if selected_streak is not None:
                    gdf = convert_sql_to_gdf(selected_streak, 'Stadium Location').sort_values("date")
                    route,distance,names_list=find_optimum_stadiums(gdf)
                    gdf_filtered = gdf[gdf['unique_id'].isin(names_list)]
                    map_html = plot_streak_map(gdf, 0.005)
                else:
                    error_message = "No data found for this streak ID."
            except ValueError:
                error_message = "Invalid Streak ID. Please enter a number." #so this error_message variable is what is piccked up by the HTML

    return render_template("index.html", map_html=Markup(map_html) if map_html else None, error_message=error_message)

if __name__ == "__main__":
    app.run(debug=True)