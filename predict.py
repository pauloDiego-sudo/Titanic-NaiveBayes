import joblib
import numpy as np
import pandas as pd

# Carregar o modelo salvo
model = joblib.load('titanic_naive_bayes_model.pkl')

# Exemplo de novos dados para predição (as mesmas features usadas no treinamento)
# Aqui estamos usando dados de exemplo com as colunas: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, FamilySize, AgeBin
# Você pode ajustar os valores conforme o seu dataset
new_data = {
    'Pclass': [1],
    'Sex': [0],           # 1 represents 'male', 0 represents 'female'
    'Age': [30],          # Add Age
    'SibSp': [0],         # Add SibSp
    'Parch': [0],         # Add Parch
    'Fare': [600],
    'Embarked': [2],      # Label encoded for 'Embarked' (adjust based on encoding)
}

# Criar um DataFrame com os novos dados
new_data_df = pd.DataFrame(new_data)

# Fazer a predição
prediction = model.predict(new_data_df)

# Exibir o resultado da predição (0 = Não Sobreviveu, 1 = Sobreviveu)
print(f'Predição: {prediction[0]}')
