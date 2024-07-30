import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import logging
from io import StringIO
from fastapi.responses import JSONResponse
import json
import torch
import faiss
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from pandas.api.types import is_object_dtype
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score

app = FastAPI()

# Global variables for model, tokenizer, index, and blog data
model = None
tokenizer = None
index = None
blog_data = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model():
    global model, tokenizer
    model_name = "distilgpt2"
    model_path = f"./models/{model_name}"
    
    if os.path.exists(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, output_hidden_states=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
        
        # Save the model locally
        os.makedirs(model_path, exist_ok=True)
        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)
    
    model.eval()  # Set the model to evaluation mode
    if torch.cuda.is_available():
        model = model.to('cuda')

def load_data():
    global index, blog_data
    index = faiss.read_index("mental_distil.index")
    with open("blog_distil.json", "r") as f:
        blog_data = json.load(f)

@app.on_event("startup")
async def startup_event():
    load_model()
    load_data()

def create_embedding(text):
    inputs = tokenizer(text, truncation=True, return_tensors="pt", max_length=512)
    if torch.cuda.is_available():
        inputs = inputs.to('cuda')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.hidden_states[-1][:, 0, :].cpu().numpy()

def process_query(query, top_k=5):
    query_embedding = create_embedding(query)[0]
    distances, indices = index.search(query_embedding.reshape(1, -1), top_k)
    return indices[0]

def generate_response(query, context):
    input_text = f"Query: {query}\nContext: {context}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=900)
    if torch.cuda.is_available():
        inputs = inputs.to('cuda')
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=124, num_beams=4, early_stopping=True)
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def query_and_respond(query):
    relevant_indices = process_query(query)
    context = " ".join([blog_data[i]['text'] for i in relevant_indices])
    response = generate_response(query, context)
    return response

class Query(BaseModel):
    text: str

# New functions for the classification endpoint
def cat_num(df, output):
    cat_columns = []
    num_columns = []
    for i in df.columns:
        if pd.api.types.is_object_dtype(df[i]) or df[i].nunique() < 8:
            cat_columns.append(i)
        else:
            num_columns.append(i)
    if output in cat_columns:
        cat_columns.remove(output)
    return cat_columns, num_columns

def preprocess_data(df, output):
    cat_columns, num_columns = cat_num(df, output)
    columns = df.columns
    transformers = [('encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_columns)]
    multicolumn_prep = ColumnTransformer(transformers, remainder='passthrough')
    df_encoded = pd.DataFrame(multicolumn_prep.fit_transform(df), columns=columns)
    
    # Label encode the target column
    label_encoder = LabelEncoder()
    df_encoded[output] = label_encoder.fit_transform(df_encoded[output])
    
    return df_encoded, multicolumn_prep, label_encoder

def train_and_compare_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_f1 = f1_score(y_test, rf_pred, average='weighted')
    
    # xgb_model = XGBClassifier(random_state=42)
    # xgb_model.fit(X_train, y_train)
    # xgb_pred = xgb_model.predict(X_test)
    # xgb_f1 = f1_score(y_test, xgb_pred, average='weighted')
    
    # if rf_f1 > xgb_f1:
    #     return rf_model, 'RandomForest', rf_f1
    # else:
    #     return xgb_model, 'XGBoost', xgb_f1

    return rf_model, 'RandomForest', rf_f1

@app.post("/classification")
async def classification_endpoint(file: UploadFile = File(...)):
    try:
        logger.info("Classification endpoint called")
        
        # Read the uploaded CSV file
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        logger.info(f"CSV file read successfully. Shape: {df.shape}")
        
        # Assume the last column is the target
        output_column = df.columns[-1]
        logger.info(f"Target column: {output_column}")
        
        # Preprocess the data
        df_encoded, preprocessor, label_encoder = preprocess_data(df, output_column)
        logger.info("Data preprocessed successfully")
        
        # Split features and target
        X = df_encoded.drop(columns=[output_column])
        y = df_encoded[output_column]
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        
        # Train and compare models
        best_model, model_name, f1_score = train_and_compare_models(X, y)
        logger.info(f"Best model: {model_name}, F1 Score: {f1_score}")
        
        # Save the model, preprocessor, and label encoder
        model_filename = f'best_model_{model_name}.joblib'
        joblib.dump({
            'model': best_model, 
            'preprocessor': preprocessor, 
            'label_encoder': label_encoder
        }, model_filename)
        logger.info(f"Model saved as {model_filename}")
        
        # return JSONResponse(content={
        #     "message": "Model trained and saved successfully",
        #     "model_name": model_name,
        #     "f1_score": f1_score
        # })
        return FileResponse(
            model_filename,
            media_type='application/octet-stream',
            filename=model_filename,
            headers={"model_name": model_name, "f1_score": str(f1_score)}
        )
    
    except Exception as e:
        logger.error(f"Error in classification endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/rag")
async def rag_endpoint(query: Query):
    try:
        response = query_and_respond(query.text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)