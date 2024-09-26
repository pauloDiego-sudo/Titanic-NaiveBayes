import joblib
import numpy as np
import pandas as pd

# Carregar o modelo salvo
model = joblib.load('titanic_naive_bayes_model.pkl')

# Exemplo de novos dados para predição (as mesmas features usadas no treinamento)
new_data = {
    'Pclass': [1],
    'Sex': [0],           
    'Age': [30],          
    'SibSp': [0],         
    'Parch': [0],         
    'Fare': [600],
    'Embarked': [2],      
}

# Criar um DataFrame com os novos dados
new_data_df = pd.DataFrame(new_data)

# Fazer a predição
prediction = model.predict(new_data_df)

# Exibir o resultado da predição (0 = Não Sobreviveu, 1 = Sobreviveu)
print(f'Predição: {prediction[0]}')
