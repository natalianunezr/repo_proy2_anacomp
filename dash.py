import dash
import math
import pandas as pd
import plotly.graph_objs as go
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

#Estilo
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# Crear la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Crear variables para cada una de las variables
edad = dcc.Input(id='input-edad', type='number', placeholder='Inserte Edad')
sex = dcc.Input(id='input-sex', type='number', placeholder='Inserte Sexo')
thal = dcc.Input(id='input-thal', type='number', placeholder='Inserte Thal')
slope = dcc.Input(id='input-slope', type='number', placeholder='Inserte Slope')
oldpeak = dcc.Input(id='input-oldpeak', type='number', placeholder='Inserte Oldpeak')
exang = dcc.Input(id='input-exang', type='number', placeholder='Inserte Exang')
ca = dcc.Input(id='input-ca', type='number', placeholder='Inserte Ca')
cp = dcc.Input(id='input-cp', type='number', placeholder='Inserte Cp')


# Crear contenedor para el título
title_container = html.Div(
    children=[        html.Div(children=[            html.H1(children='Predictor de enfermedades cardíacas - Home Test', style={'text-align': 'center'})        ], style={'background-color': '#6495ED', 'padding': '10px', 'border-radius': '5px'})
    ]
)

subtitle_container = html.Div(
    children=[
        html.H2(children='Responda en cada una de las siguientes casillas con sus datos')
    ]
)

# Crear contenedor para los inputbox
input_container = html.Div(
    children=[
        html.Table([
            html.Thead([
                html.Tr([
                    html.Th("Variable"),
                    html.Th("Valor"),
                    html.Th("Instrucciones")
                ])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td("Edad"),
                    html.Td(edad),
                    html.Td("Ingrese su edad en años")
                ]),
                html.Tr([
                    html.Td("Sexo"),
                    html.Td(sex),
                    html.Td("Ingrese su sexo formato binario (M=1/F=0)")
                ]),
                html.Tr([
                    html.Td("Thal"),
                    html.Td(thal),
                    html.Td("Ingrese el tipo de defecto talámico (3/6/7)")
                ]),
                html.Tr([
                    html.Td("Slope"),
                    html.Td(slope),
                    html.Td("Ingrese el tipo de la pendiente del segmento ST (0/1/2)")
                ]),
                html.Tr([
                    html.Td("Oldpeak"),
                    html.Td(oldpeak),
                    html.Td("Ingrese la depresión del segmento ST inducida por el ejercicio en relación con el reposo (0.0 a 6.2)")
                ]),
                html.Tr([
                    html.Td("Exang"),
                    html.Td(exang),
                    html.Td("Ingrese si hay presencia de angina inducida por ejercicio (0 = no, 1 = si)")
                ]),
                html.Tr([
                    html.Td("Ca"),
                    html.Td(ca),
                    html.Td("Ingrese el número de vasos principales coloreados por flourosopía (0/1/2/3)")
                ]),
                html.Tr([
                    html.Td("Cp"),
                    html.Td(cp),
                    html.Td("Ingrese el tipo de dolor torácico (0/1/2/3)")
                ])
            ])
        ])
    ]
)

# Crear contenedor para el botón
button_container = html.Div(
    children=[
        html.Br(),
        html.Button('Evaluar riesgo de enfermedad', id='boton_calcular', n_clicks=0),
        html.Br(), # Agregar espacio vacío
        html.Div(id="resultado") # Aquí se mostrará el resultado
    ]
)


#Importamos los datos como dataframe
columnas = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
df = pd.read_csv('processed_cleveland.data', names=columnas)

#Convertimos datos a números

#En donde tengamos datos faltantes (?) ponemos 0

for column in df.columns:
    # Reemplazar valores "?" con 0 en la columna y fila respectivas
    df[column] = df[column].replace('?', 0)

df = df.apply(pd.to_numeric, errors='coerce')    

#Discretizar variables
#La variable age tiene que ser discreta para la red bayesiana, por lo que la dividiremos por cuartiles siendo 29 la edad minima y 77 la edad maxima 
df['age_discreta'] = df['age']
df.loc[df['age_discreta'] < (df['age_discreta'].describe()['25%']), 'age'] = 1
df.loc[((df['age_discreta'].describe()['25%']) <= df['age_discreta']) & (df['age_discreta'] < (df['age_discreta'].describe()['50%'])), 'age'] = 2
df.loc[((df['age_discreta'].describe()['50%']) <= df['age_discreta']) & (df['age_discreta'] < (df['age_discreta'].describe()['75%'])), 'age'] = 3
df.loc[((df['age_discreta'].describe()['75%']) <= df['age_discreta']) & (df['age_discreta'] < 77), 'age'] = 4
df.age = df.age.astype(int)

#La variable oldpeak tiene que ser discreta, por lo que la dividiremos por cuartiles siendo 0 el cuartil minimo y 6.2 el cuartil maximo
df['oldpeak_discreta'] = df['oldpeak']

df.loc[df['oldpeak_discreta'] < 0.8, 'oldpeak' ] = 1
df.loc[(0.8 <= df['oldpeak_discreta']) & (df['oldpeak_discreta'] < 1.6), 'oldpeak'] = 2
df.loc[(1.6 <= df['oldpeak_discreta']) & (df['oldpeak_discreta'] < 6.2), 'oldpeak'] = 3

df.oldpeak = df.oldpeak.astype(int)

#Variable num
df.loc[(df['num'] == 0) , 'num_discreta'] = 0
df.loc[(df['num'] != 0) , 'num_discreta'] = 1
df.num = df.num.astype(int)

#Eliminamos columnas sobrantes
df.drop(['num', 'age_discreta', 'oldpeak_discreta'], axis=1, inplace=True)
#df


#se crea la red bayesiana
#Creamos modelo
model = BayesianNetwork(
    [ ("age","num_discreta"),
     ("sex","num_discreta"),
     ("num_discreta","thal"),
     ("num_discreta","exang"),
     ("num_discreta","slope"),
     ("num_discreta","cp"),
     ("num_discreta","oldpeak"),
     ("num_discreta","ca"),

    ]

)
#parametrizacion
model.fit(
    data=df,
    estimator=MaximumLikelihoodEstimator
)
#discretizacion
# ---------------------------------------------
# Definir función para calcular la probabilidad
#----------------------------------------------

# Con el fin de discretizar la variable age , utilizamos para el 25% 48 anos, para el 50% 56 anos y para el 75% 61 anos.
def discretizar_age(age):
    if age<48:
        return 1
    elif age<56:
        return 2
    elif age<61:
        return 3
    else:
        return 4

def discretizar_oldpeak(oldpeak):
    if oldpeak<0.8:
        return 1
    elif oldpeak<1.6:
        return 2
    else:
        return 3
    
def inference(evidence):
    infer=VariableElimination(model)
    prob=infer.query(variables=['num_discreta'], evidence=evidence)
    return prob.values.tolist()

# NO SE QUE ES ESTO--------------------------------------------------------------------------------------------------------------------
# Agregar todo al layout
app.layout = html.Div(children=[
    title_container,
    subtitle_container,
    input_container,
    button_container
])
#Callbacks para conectar botón con función
@app.callback(
    Output('resultado', 'children'),
    Input('boton_calcular', 'n_clicks'),
    [Input('input-edad', 'value'),
     Input('input-sex', 'value'),
     Input('input-thal', 'value'),
     Input('input-slope', 'value'),
     Input('input-oldpeak', 'value'),
     Input('input-exang', 'value'),
     Input('input-ca', 'value'),
     Input('input-cp', 'value')]
)


def actualizar_output(n_clicks, input_edad, input_sex, input_thal, input_slope, input_oldpeak, input_exang, input_ca, input_cp):
    if n_clicks is not None:
        resultado = calcular_probabilidad(input_edad, input_sex, input_thal, input_slope, input_oldpeak, input_exang, input_ca, input_cp)
        return resultado

# Correr la aplicación Dash
if __name__ == '__main__':
    app.run_server(debug=True)
