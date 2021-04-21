import pickle
import requests
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt


#data preparation
dados = pd.read_csv('titanic.csv')

a_renomear = {
    'Survived': 'sobreviveu',
    'Pclass': 'classe',
    'Sex': 'sexo',
    'Siblings/Spouses Aboard': 'familiares_02',
    'Parents/Children Aboard': 'familiares_01',
    'Fare': 'tarifa'
}

dados = dados.rename(columns=a_renomear)

a_trocar = {
    'male' : 0,
    'female' : 1
}

dados.sexo = dados.sexo.map(a_trocar)

#export data
pickle.dump(dados.sexo, open('parametros/sexo.pkl', 'wb'))



#data separation
y = dados['sobreviveu']
x = dados[['classe', 'sexo', 'familiares_02', 'familiares_01', 'tarifa']]

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.25, stratify=y)
print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))


#Scalling the data for SVC
scaler = StandardScaler()
scaler.fit(treino_x)
treino_x = scaler.transform(treino_x)
teste_x = scaler.transform(teste_x)


#model creation and training
modelo = SVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

#accuracy
acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia foi %.2f%%" % acuracia)

#prediction
pred = modelo.predict(teste_x)
rmse = np.sqrt(mean_squared_error(teste_y, pred))
mae = mean_absolute_error(teste_y, pred)
r2 = r2_score(teste_y, pred)

print('RMSE: {}'.format(rmse))
print('MAE: {}'.format(mae))
print('R2: {}'.format(r2))

# save trained model
pickle.dump(modelo, open('modelo/modelo-titanic.pkl', 'wb'))

#request test
df_json = x.to_json(orient='records')

url = 'https://titanic-survivor-model.herokuapp.com/predict'
data = df_json
header = {'Content-type': 'application/json'}

r = requests.post(url=url, data=data, headers=header)

#output test
print(r)
print(r.json())

