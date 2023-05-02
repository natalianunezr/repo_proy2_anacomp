import dash
from dash import html
import dash_bootstrap_components as dbc

index = html.Div(children= [

    #Título
    html.H1(children = '''Bienvenido a la app de testeo remoto''', style = {'textAlign': 'center'}),
    html.Br(),
    
#Vínculos para ir a la otra página    
    html.Div(children = [
        html.Div(children = [
            dbc.Button("Visualizaciones", size = "lg", id = "visualizaciones", href = "/visu", style = {'margin-right':'20px', 'verticalAlign': 'middle'})],

        style = {'margin_bottom': '20px',
                'display': 'flex',
                'justify-content': 'center'}),
    html.Br(),


])
    
    
    




])
