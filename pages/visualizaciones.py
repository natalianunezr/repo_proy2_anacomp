import psycopg2
import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

dash.register_page(__name__)

# Establecer la conexión a la base de datos
conn = psycopg2.connect(
    host="proyectodos.cuz8vbzi7nnr.us-east-1.rds.amazonaws.com",
    database="datos_finales",
    user="postgres",
    password="proyectodos"
)

# Crear un objeto cursor a partir de la conexión
cursor = conn.cursor()

# Ejecutar una consulta SQL
query = "SELECT * FROM usuarios;"
cursor.execute(query)

# Obtener los resultados y cargarlos en un DataFrame
df = pd.read_sql(query, conn)

# Cerrar el cursor y la conexión
cursor.close()
conn.close()

#Grafica 1
df_filtered = df[(df['oldpeak'] >= 1) & (df['oldpeak'] <= 3)]
df_filtered['sex'] = df_filtered['sex'].map({1:'Hombre', 0:'Mujer'})
colors = {"Hombre": "#2e5266", "Mujer": "#5e96ae"}


fig = px.scatter(df_filtered, x="oldpeak", y="num_discreta", size="ca", color="sex", color_discrete_map=colors, 
                 title="Relación entre el nivel de depresión en el ejercicio y enfermedad cardiaca",
                 labels={"oldpeak": "Nivel de depresión en el ejercicio", "num_discreta": "Enfermedad cardiaca",
                         "sex": "Sexo"},size_max=10
)
fig.update_traces(mode='markers', marker=dict(sizemode='diameter'))
fig.update_layout(title="Relación entre el nivel de depresión en el ejercicio y enfermedad cardiaca", title_x=0.5, title_font=dict(size=24))
fig.update_layout(yaxis_title='Enfermedad cardiaca')
fig.update_layout(xaxis_title='Nivel de depresión en el ejercicio')
fig.update_xaxes(tickvals=['1', '2', '3'], ticktext=['Menor a 0.8', '0.8-1.6', 'Mayor a 1.6'])
fig.update_yaxes(tickvals=['1', '0'], ticktext=['Tiene la enfermedad', 'No tiene la enfermedad'])


#Grafica 2
df["cp"] = df["cp"].replace({1:"Angina tipica", 2:"Angina atipica", 3:"Dolor no aginal", 4:"Asintomatico"})
df_filtered2 = df[df['thal'].isin([3, 6, 7])]
colors2 = {"Angina tipica": "#97c1e3", "Angina atipica": "#5e96ae", "Dolor no aginal": "#2e5266", "asintomatico": "#152238"}

fig2 = px.histogram(df_filtered2, x="thal", color="cp",color_discrete_map=colors2, facet_col="sex", 
                   facet_col_wrap=2, 
                   labels={"thal":"Tipo de defecto cardiaco", "cp":"Tipo de dolor de pecho", "sex":"Género"})

fig2.update_layout(bargap=0.2)
fig2.update_layout(title="Distribución de Thal por CP y Género", title_x=0.5, title_font=dict(size=24))
fig2.update_xaxes(tickvals=['3', '6 ', '7'], ticktext=['Normal', 'Corregido', 'Reversible'])
fig2.update_layout(yaxis_title='Frecuencia de defecto cardiaco')


#Grafica 3
df_filtered3 = df[(df['age'] >= 1) & (df['age'] <= 4)]
df_grouped = df_filtered3.groupby('age')['num_discreta'].mean()
# Crear el histograma
fig3 = px.histogram(df_grouped, x=df_grouped.index, y=df_grouped.values, color="num_discreta", 
                   barmode='overlay', nbins=len(df_grouped), labels={'x':'Edad', 'y':'Porcentaje de personas enfermas'},color_discrete_sequence=['#97c1e3', '#5e96ae', '#2e5266', '#152238'])
fig3.update_layout(title="Histograma de porcentaje de personas enfermas por edad", title_x=0.5, title_font=dict(size=24))
fig3.update_xaxes(tickvals=['1', '2', '3', '4'], ticktext=['29-48', '48-56', '56-61', '61-77'])
fig3.update_layout(yaxis_tickformat = ',.0%')
fig3.update_layout(yaxis_title='Porcentaje de personas enfermas')
fig3.update_layout(xaxis_title='Edad')
#Crear pagina del dash para las tres graficas

dash.register_page(__name__)
layout = html.Div([html.H1("Gráficas de enfermedades del corazón"),
                   dcc.Graph(id='bar-fig', figure=fig),
                   dcc.Graph(id='histogram', figure=fig2),
                   dcc.Graph(id='histogram', figure=fig3)])
