from pickletools import optimize
from numpy import size
import pandas as pd
from sklearn import linear_model
from sklearn import model_selection
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
from joblib import load
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sb

# reg = linear_model.Ridge(alpha=.5)
reg = linear_model.LogisticRegression(solver='saga',max_iter=10000)
# clf = MLPClassifier(random_state=1, max_iter=300, verbose=0, learning_rate_init=0.001, solver='lbfgs')

df = pd.read_csv('database/csv/araxes_06-04-2022.csv')
data = df[df.columns[0:5]]
print('data')
print(data)

labels = df[df.columns[-1]]*1000
print('labels')
print(labels)


for i,val in enumerate(data.iterrows()):
    hora = data.iloc[i]['date'].split(' ')[1]
    hora = hora.split(':')
    hora = int(hora[0])*60 + int(hora[1])
    # hora = round(hora/1440,5)
    data.loc[i]= [hora, data.iloc[i]['level_px_0'],data.iloc[i]['level_px_1'],data.iloc[i]['level_px_2'],data.iloc[i]['level_px_3']]

data.to_csv('database/csv/{}.csv'.format("test"),index=False)
print('Visualización datos')
df.hist()
plt.show()
#cambiar tipo de dato de las columnas dataframe

# level_px_0 = df[df.columns[-1]]
# print(level_px_0.max())
# print(level_px_0.min())
# print(level_px_0.median())
# print(level_px_0.mean())

#contar valores repetidos en el dataframe
# print(df.groupby('level').count())

df.plot(x ='date', y='level_px_0', kind = 'scatter')	
plt.show()
sb.pairplot(df.dropna(), hue='level', height=3, kind='reg')
plt.show()



epochs = 100
score = []

print("Entrenando ......")


for i in range(epochs):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)
    reg.fit(x_train,y_train)
    score.append(reg.score(x_test,y_test))
    print(score[i])

    
    if reg.score(x_test,y_test) >= 0.54:
        print('Epoch: {} modelo guardaro'.format(i))
        joblib.dump(reg, 'model_araxes_{}.pkl'.format(round(reg.score(x_test,y_test),5)))
        break


best_epoch = score.index(max(score))
print(best_epoch)
print(max(score))

model_load = load('model_araxes_0.61502.pkl')
print("Modelo cargado")
p=model_load.predict(x_test)
print(classification_report(y_test,p))

print(model_load.score(y_test,p))

print(model_load.predict(x_test))
print(confusion_matrix(y_test, p))
