import pandas as pd

data = pd.read_csv('diabetes.csv')

from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator

model = BayesianModel([('Pregnancies', 'Outcome'),
                       ('Glucose', 'Outcome'),
                       ('BloodPressure', 'Outcome'),
                       ('SkinThickness', 'Outcome'),
                       ('Insulin', 'Outcome'),
                       ('BMI', 'Outcome'),
                       ('DiabetesPedigreeFunction', 'Outcome'),
                       ('Age', 'Outcome')])

model.fit(data, estimator=BayesianEstimator)

from pgmpy.inference import VariableElimination

inference = VariableElimination(model)  
pred = inference.query(['Outcome'], evidence={'Pregnancies': 6, 
                                                   'Glucose': 148,
                                                   'BloodPressure': 72,
                                                   'SkinThickness': 35,
                                                   'Insulin': 0,
                                                   'BMI': 33.6,
                                                   'DiabetesPedigreeFunction': 0.627,
                                                   'Age': 50,
                                                   })
print(pred)