# 🥔 Potato Plant Disease Classifier

A machine learning application that identifies diseases in potato plant leaves using deep learning.

## 📝 Description

This project uses TensorFlow to classify potato plant leaf images into three categories:
- Early Blight
- Late Blight
- Healthy

The application provides a user-friendly web interface built with Streamlit, making it easy for farmers and agricultural professionals to diagnose potato plant diseases by simply uploading a photo.

## 🛠️ Technologies Used

- **TensorFlow**: For the deep learning model
- **Streamlit**: For the web interface
- **Python**: Programming language
- **PIL/Pillow**: For image processing
- **NumPy**: For numerical operations
- **FastAPI**: Used in development for API creation

## 🔧 Installation & Setup

1. Clone the repository
   ```
   git clone https://github.com/YourUsername/Potato-plant-disease.git
   cd Potato-plant-disease
   ```

2. Create and activate a virtual environment (Optional but recommended)
   ```
   # Using conda
   conda create -n potato-env python=3.10
   conda activate potato-env
   
   # OR using venv
   python -m venv potato-env
   # On Windows
   potato-env\Scripts\activate
   # On macOS/Linux
   source potato-env/bin/activate
   ```

3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

4. Run the application
   ```
   streamlit run main.py
   ```

5. Open your browser and go to `http://localhost:8501`

## 📊 Dataset

The model was trained on a dataset of potato leaf images categorized into three classes:
- Potato___Early_blight
- Potato___Late_blight
- Potato___Healthy

Each category contains multiple images that were used to train the model to recognize the visual patterns associated with different conditions.

## 🚀 Usage

1. Launch the application using `streamlit run main.py`
2. Upload an image of a potato plant leaf through the web interface
3. The model will process the image and display:
   - The predicted disease class
   - The confidence level of the prediction

## 📁 Project Structure

```
Potato-plant-disease/
├── main.py                # Streamlit web application
├── training.ipynb         # Jupyter notebook for model training
├── requirements.txt       # Project dependencies
├── models/                # Saved TensorFlow models
├── Potato-plant-disease/  # Dataset directory
│   ├── Potato___Early_blight/
│   ├── Potato___Late_blight/
│   └── Potato___Healthy/
└── README.md              # Project documentation
```

## 🔜 Future Improvements

- Add more plant diseases to the classifier
- Implement model explainability to highlight areas of the leaf that influenced the prediction
- Create a mobile app version for field use
- Add treatment recommendations based on detected diseases
- Improve model accuracy with more diverse training data


## 🙏 Acknowledgments

- The dataset used for training this model
- TensorFlow and Streamlit communities for their excellent documentation

