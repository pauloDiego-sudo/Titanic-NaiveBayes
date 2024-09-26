from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware

# Carregando o modelo e os Encoders
loaded_model = joblib.load('titanic_naive_bayes_model.pkl') 
label_encoder_sex = joblib.load('label_encoder_sex.pkl')  
label_encoder_embarked = joblib.load('label_encoder_embarked.pkl') 

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definindo o modelo de entrada Pydantic
class Passenger(BaseModel):
    pclass: int
    sex: str
    age: float
    sibsp: int
    parch: int
    fare: float
    embarked: str

# Função de previsão 
def predict_survival(pclass, sex, age, sibsp, parch, fare, embarked):
    sex_encoded = label_encoder_sex.transform([sex])[0]
    embarked_encoded = label_encoder_embarked.transform([embarked])[0]
    input_data = [[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]]
    prediction = loaded_model.predict(input_data)[0]
    return 'Sobreviveu' if prediction == 1 else 'Não sobreviveu'

# Rota /predict
@app.post("/predict")
def predict(passenger: Passenger):
    prediction = predict_survival(passenger.pclass, passenger.sex, passenger.age, 
                                  passenger.sibsp, passenger.parch, passenger.fare, 
                                  passenger.embarked)
    return {"prediction": prediction}

# Rota /survived 
@app.get("/survived")
def survived_passengers():
    return {"message": "Esta funcionalidade ainda não foi implementada. Retornará uma lista de passageiros que sobreviveram."} 

# Rota /died
@app.get("/died")
def died_passengers():
    return {"message": "Esta funcionalidade ainda não foi implementada. Retornará uma lista de passageiros que morreram."}