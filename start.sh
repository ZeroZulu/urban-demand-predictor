#!/bin/bash
# Download model files if not present
if [ ! -f "models/best_model.pkl" ]; then
    echo "Downloading model files..."
    mkdir -p models
    curl -L "https://github.com/user-attachments/files/26249314/models_v1.zip" -o models/models_v1.zip
    cd models && unzip -o models_v1.zip && cd ..
    echo "Models downloaded and extracted."
else
    echo "Models already present."
fi

# Start the API
uvicorn api.main:app --host 0.0.0.0 --port $PORT
