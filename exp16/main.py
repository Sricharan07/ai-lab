import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.models import BayesianModel

model = BayesianModel([('X', 'Y')]) 


data = pd.DataFrame({'X': [1, 2, 3, 4, 5], 'Y': [10, 20, 30, 40, 50]})

model.fit(data, estimator=BayesianEstimator)

pdf_X = model.get_cpds('X')
pdf_Y = model.get_cpds('Y')

print("PDF of X:")
print(pdf_X)
print("\nPDF of Y:")
print(pdf_Y)
