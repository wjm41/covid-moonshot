import scipy.optimize as optimization
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score
import numpy as np 

def hill_eqn(x, a, b, c, d):
    return d + (a - d) / (1 + np.power(10,(x - c)*b))

def return_IC50(x,y):
    # x should be in log10 form
    errors = np.ones_like(x)

    # initial parameter guess for Hill equation - pIC50 as -5.0
    x0 = np.array([100, 1.5, -4.3, 0.0])

    # least-sq fitting, returns optimal parameters as well as chi-sq etc
    opt_results = optimization.curve_fit(hill_eqn, x, y, x0, errors)
    return opt_results[0][2] # this is param C ie the pIC50
import pandas as pd
df = pd.read_csv('~/Downloads/AID_1890_datatable_all.csv')[5:]
#print(df)

df_inhib = df[['Inhibition at 3.0 nM','Inhibition at 9.1 nM','Inhibition at 27.3 nM','Inhibition at 81.8 nM','Inhibition at 245.4 nM','Inhibition at 736.3 nM','Inhibition at 2.2 uM','Inhibition at 6.6 uM','Inhibition at 19.9 uM','Inhibition at 59.6 uM']]

y_test = df['LogIC50'].astype(float).values
y_data = []
for i, row in df_inhib.iterrows():
     y_data.append(row.values)
#print(y_data)
x_data = [3E-9, 9.1E-9, 27.3E-9, 81.8E-9, 245.4E-9, 736.3E-9, 2.2E-6, 6.6E-6, 19.9E-6, 59.6E-6]
x_data = np.log10(x_data)

y_pred = []
for y in y_data:
    y_pred.append(return_IC50(x_data, y))

for i in range(len(y_pred)):
    print(np.power(10.0,y_pred[i]), np.power(10.0,y_test[i]))
#print(np.power(10, y_pred))
#print(np.power(10, y_test))
print('R2 = {:.3f}'.format(r2_score(y_test, y_pred)))
print('RMSE = {:.3f}'.format(np.sqrt(mean_squared_error(y_test, y_pred))))
#y_data = [2.7, 4.9, 4.8, 5.8, 6.4, 9.1, 17.1, 41.4, 65.3, 82.7]
#y_data = [-1.1, -1.7, -1.1, -0.2, 1.1, -192.9, -227.1, -47.1, 28.9, 48.1]
#y_data = [0.2,0.2,-0.6,2.5,14.3,37,71.2,91.3,97.1,98.7]

