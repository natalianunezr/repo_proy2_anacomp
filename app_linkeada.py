import dash
import math
import pandas as pd
import plotly.graph_objs as go
from dash import html
from dash import dcc


#Estilo
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# Crear la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, use_pages=True)

# Crear contenedor para el título
app.layout = html.Div(
    children=[        html.Div(children=[            
        html.H1(children='Bienvenido a la prueba desde casa sobre riesgo de enfermedades cardíacas', style={'font-weight': 'bold','fontSize':50, 'text-align': 'center'})        ], 
        style={'background-color': '#dfe6f2', 'padding': '30px', 'border-radius': '5px'}),
        html.Div([
            dcc.Link(page['name']+" | ", href=page['path'])
            for page in dash.page_registry.values()
        ]),
        html.Hr(),

        #Contenido de cada page
        dash.page_container
    ]
)


if __name__ == "__main__":
    app.run(debug=True)