#--------------------------------------
#---------EVALUACIÓN DE MODELOS--------
#--------------------------------------

#Importo librerías
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel
from pgmpy.sampling import BayesianModelSampling
import math
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#Pongo mi modelo

#Importamos los datos como dataframe
columnas = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
df = pd.read_csv('datos.data', names=columnas)

#Discretizar num
df.loc[(df['num'] == 0) , 'num_discreta'] = 0
df.loc[(df['num'] != 0) , 'num_discreta'] = 1
df.num = df.num.astype(int)

#Defino mis datos de entrenamiento 
data_train = df.iloc[1:75, :]
data_test = df.iloc[75:, :]

#Modelo a entrenar con datos de entrenamiento
model = BayesianNetwork(
    [ ("age","num"),
     ("sex","num"),
     ("num","thal"),
     ("num","exang"),
     ("num","slope"),
     ("num","cp"),
     ("num","oldpeak"),
     ("num","ca"),

    ]

)
model.fit(data_train, estimator=MaximumLikelihoodEstimator)

#Miro probabilidades del modelo usando evidencia datos de entrenamiento
lista_mio =[]

inferencia_mio = VariableElimination(model)

for i in range(1,len(data_train)):

    proba_mio = inferencia_mio.query(variables=data_train['num'], evidence={'age': data_train["age"][1],'sex': data_train["sex"][2], 'cp': data_train["cp"][3],
    'exang': data_train["exang"][9], 'oldpeak': data_train["oldpeak"][10],
    'slope': data_train["slope"][11], 'ca': data_train["ca"][12], 'thal': data_train["thal"][13]})

    if proba_mio.values[0] <= 0.5:
        lista_mio.append(0) #no
    else:
        lista_mio.append(1) #si

#matriz de confusion
df_confu = pd.DataFrame({'actual': df["num"], 'predicted': lista_mio})
cm = confusion_matrix(df_confu['actual'], lista_mio)
print(cm)


