# Import required libraries
import pandas as pd
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.express as px

# Read the airline data into pandas dataframe
spacex_df = pd.read_csv("spacex_launch_dash.csv")
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()

# Create a dash application
app = dash.Dash(__name__)

# Create an app layout
app.layout = html.Div(children=[html.H1('SpaceX Launch Records Dashboard',
                                        style={'textAlign': 'center', 'color': '#503D36',
                                               'font-size': 40}),
                                # TASK 1: Add a dropdown list to enable Launch Site selection
                                # The default select value is for ALL sites
                                

                                dcc.Dropdown(
                                    id="site-dropdown",
                                    options =[
                                    {"label": "All Sites", "value": "ALL"},
                                    {"label": "site1", "value": "site1"},
                                    {"label": "site2", "value": "site2"},
                                    {"label": "site3", "value": "site3"},
                                    {"label": "site4", "value": "site4"},
                                    ],
                                    value="ALL",
                                    placeholder="Select a Launch Site here",
                                    searchable=True
                                ),
                                html.Br(),

                                # TASK 2: Add a pie chart to show the total successful launches count for all sites
                                # If a specific launch site was selected, show the Success vs. Failed counts for the site
                                html.Div(dcc.Graph(id='success-pie-chart')),
                                html.Br(),

                                html.P("Payload range (Kg):"),
                                # TASK 3: Add a slider to select payload range
                                dcc.RangeSlider(
                                    id="payload-slider",
                                    min=min_payload, max=max_payload, step=1000,
                                    marks={i: f"{i} kg" for i in range(0, 11000, 2500)},
                                    value=[min_payload, max_payload]
                                ),

                                html.Br(),
                                # TASK 4: Add a scatter chart to show the correlation between payload and launch success
                                html.Div(dcc.Graph(id='success-payload-scatter-chart')),
                                ])

# TASK 2:
# Add a callback function for `site-dropdown` as input, `success-pie-chart` as output
@app.callback(
    Output("success-pie-chart", "figure"),
    Input("site-dropdown", "value")
)
def get_pie_chart(selected_site):
    """Render a pie chart for success rate based on the selected launch site."""
    if selected_site == "ALL":
        # Pie chart for all sites (total success launches per site)
        fig = px.pie(spacex_df, values="class", names="Launch Site", title="Total Successful Launches")
    else:
        # Pie chart for the selected site (Success vs Failure)
        filtered_df = spacex_df[spacex_df["Launch Site"] == selected_site]
        fig = px.pie(filtered_df, names="class", title=f"Launch Outcome for {selected_site}")

    return fig


# TASK 4:
# Add a callback function for `site-dropdown` and `payload-slider` as inputs, `success-payload-scatter-chart` as output
@app.callback(
    Output("success-payload-scatter-chart", "figure"),
    [Input("site-dropdown", "value"),
     Input("payload-slider", "value")]
)
def get_scatter_chart(selected_site, payload_range):
    """Render a scatter plot for payload mass vs. success rate based on the selected site and payload range."""
    min_val, max_val = payload_range
    filtered_df = spacex_df[(spacex_df["Payload Mass (kg)"] >= min_val) & (spacex_df["Payload Mass (kg)"] <= max_val)]

    if selected_site != "ALL":
        filtered_df = filtered_df[filtered_df["Launch Site"] == selected_site]

    # Scatter plot for Payload vs Success, colored by Booster Version
    fig = px.scatter(filtered_df, x="Payload Mass (kg)", y="class", color="Booster Version",
                     title="Payload vs Launch Outcome")
    
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(port=8055)
