# Titanic Survival Prediction API

This project implements a machine learning model to predict survival on the Titanic and exposes the model through a FastAPI application.

## Setup

1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training the Model

1. Ensure you have the Titanic dataset (`train.csv`) in the project root directory.
2. Run the training script:
   ```bash
   python training.py
   ```
   This will create the following files:
   - `titanic_naive_bayes_model.pkl`: The trained model
   - `label_encoder_sex.pkl`: Label encoder for sex feature
   - `label_encoder_embarked.pkl`: Label encoder for embarked feature

## Running Predictions

To test the model with sample data:

1. Ensure you have the model and encoder files in the project root.
2. Run the prediction script:
   ```bash
   python predict.py
   ```

## Running the API

1. Start the FastAPI application:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8080
   ```
2. The API will be available at [http://localhost:8080](http://localhost:8080)

## Using Docker

To run the application in a Docker container:

1. Build the Docker image:
   ```bash
   docker build -t titanic-api .
   ```
2. Run the Docker container:
   ```bash
   docker-compose up
   ```

## API Endpoints

- **POST** `/predict`: Predict survival for a single passenger
- **GET** `/survived`: List of passengers who survived (not implemented)
- **GET** `/died`: List of passengers who did not survive (not implemented)

## Making Predictions via API

Send a POST request to `/predict` with passenger data in JSON format:

### Example Request:

```json
{
  "pclass": 1,
  "sex": "female",
  "age": 30,
  "sibsp": 0,
  "parch": 0,
  "fare": 600,
  "embarked": "C"
}
```

### Using `curl`:

```bash
curl -X POST "http://localhost:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "pclass": 1,
           "sex": "female",
           "age": 30,
           "sibsp": 0,
           "parch": 0,
           "fare": 600,
           "embarked": "C"
         }'
```

### Example Response:

```json
{
  "prediction": "Sobreviveu"
}
```

### Using Python:

```python
import requests

url = "http://localhost:8080/predict"
data = {
    "pclass": 1,
    "sex": "female",
    "age": 30,
    "sibsp": 0,
    "parch": 0,
    "fare": 600,
    "embarked": "C"
}

response = requests.post(url, json=data)
print(response.json())
# Output: {'prediction': 'Sobreviveu'}
```

### Using JavaScript (Fetch API):

```javascript
fetch('http://localhost:8080/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    pclass: 1,
    sex: 'female',
    age: 30,
    sibsp: 0,
    parch: 0,
    fare: 600,
    embarked: 'C'
  })
})
  .then(response => response.json())
  .then(data => console.log(data));
  // Output: {prediction: 'Sobreviveu'}
```

### Using HTTPie:

```bash
http POST http://localhost:8080/predict pclass=1 sex='female' age=30 sibsp=0 parch=0 fare=600 embarked='C'
```

## Additional Examples

### Example 1: Male Passenger in Third Class

```json
{
  "pclass": 3,
  "sex": "male",
  "age": 25,
  "sibsp": 0,
  "parch": 0,
  "fare": 7.25,
  "embarked": "S"
}
```

**Expected Response:**

```json
{
  "prediction": "Não Sobreviveu"
}
```

### Example 2: Female Child in Second Class

```json
{
  "pclass": 2,
  "sex": "female",
  "age": 8,
  "sibsp": 1,
  "parch": 1,
  "fare": 29.00,
  "embarked": "Q"
}
```

**Expected Response:**

```json
{
  "prediction": "Sobreviveu"
}
```


