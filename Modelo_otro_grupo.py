import pandas as pd
import numpy as np
import os
from pgmpy.models import BayesianModel
from pgmpy.sampling import BayesianModelSampling
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import VariableElimination



a="heart_disease_modified.csv"
df = pd.read_csv(a)

model = BayesianModel([('age','chol'),('age','fbs'),('sex','chol'),('sex','fbs')
                      ,('chol','num'),('fbs','num')
                      , ('num','cp'), ('num','exang'), ('num','anom_thalach'), ('num','trestbps')])

model.fit(df, estimator=MaximumLikelihoodEstimator)

for i in model.nodes():
    print(model.get_cpds(i))

#PRUEBA INFERENCIA

edad="0-45"
sexo=0
print("PRUEBA")
#PRUEBA INFERENCIA - Ya separdo el valor, para poder hacer un IF con la recomendacion
infer = VariableElimination(model)
P1 = infer.query(['num'], evidence={'age': edad,'sex':sexo, 'chol':'240-260','fbs':1})
PROBA= P1.values
print(PROBA[0])
