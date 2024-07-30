# FastAPI Application for rag and classification model generation

This guide provides instructions to set up and run the FastAPI application for text generation and classification model training. Follow the steps below to clone the repository, download necessary files, and start the application.

## Setup Instructions

### Step 1: Clone the Repository
First, clone the repository from GitHub:

### Step 2: Download Required Files
Next, download the `blog_distil.json` and `mental_distil.index` files and place them in the project directory. You can find these files in the this repo.

### Step 3: Install Dependencies
Install the required dependencies using pip:
```
io
fastapi
uvicorn
torch
faiss-cpu
pydantic 
sckit-learn
```

### Step 4: Run the Application
Start the FastAPI application using Uvicorn:
```bash
uvicorn app:app --reload
or just run the app_2.py file it will automatically start the server
```
This will start the server at `127.0.0.1:8000`.

### Step 5: Access the API Documentation
Open your browser and navigate to `http://127.0.0.1:8000/docs` to access the API documentation. Here you will find interactive documentation where you can test the API endpoints.

### Step 6: Test the Classification Endpoint
1. Click on the `POST /classification` endpoint to expand it.
2. Click on the "Try it out" button.
3. Click on "Choose File" and upload the dataset CSV file.
4. Click the "Execute" button.
5. After execution, you will receive a response with a message indicating that the model has been trained and saved successfully. The response will include the model name and F1 score.
6. You can download the generated model file.

### Example Screenshot
![docs_1](https://github.com/user-attachments/assets/69628395-054e-426d-9dbb-809ff0040315)
![docs_3](https://github.com/user-attachments/assets/052ab8f8-8415-4f3e-8d32-ff93962f58c7)
![docs_2](https://github.com/user-attachments/assets/2b4a7d95-497f-496e-b9b4-ea864032ba91)
