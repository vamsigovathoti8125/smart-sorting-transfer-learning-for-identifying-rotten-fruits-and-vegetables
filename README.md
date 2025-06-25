smart-sorting-transfer-learning-for-identifying-rotten-fruits-and-vegetables

---


🍎 Smart Sorting: Rotten vs Fresh Fruit & Vegetable Classification Using Transfer Learning

Smart Sorting is a deep learning-powered web application designed to classify microscopic images of fruits and vegetables into fresh or rotten categories across 16 classes:

🍎 Fresh Apple, Fresh Banana, Fresh Tomato...

🥀 Rotten Apple, Rotten Banana, Rotten Tomato...


This intelligent food sorting tool leverages Transfer Learning with MobileNetV2 to deliver accurate, real-time predictions through a modern Flask-based web interface.


---

🚀 How It Works

1. 📤 Upload an image of a fruit or vegetable.


2. 🤖 Model processes the image using deep learning.


3. 🧾 Prediction is displayed along with a preview of the uploaded image.



This makes Smart Sorting ideal for food sorting systems, supply chains, quality control in agriculture, and academic research.


---

✅ Features

✅ Real-time classification of 16 fruit/vegetable classes

✅ Preprocessing pipeline using OpenCV

✅ Lightweight MobileNetV2 model for fast inference

✅ Modern UI with intuitive feedback

✅ Base64 image preview for smooth rendering



---

🛠 Tech Stack

Layer	Technologies Used

Model	TensorFlow / Keras with MobileNetV2
Backend	Python, Flask
Image Preprocessing	OpenCV
Frontend	HTML5, CSS3 (Modern, clean UI)

---

🧪 Run Locally

You can run this project on your local system by following these simple steps:

1️⃣ Clone the Repository

git clone https://github.com/yourusername/SmartSortingApp.git
cd SmartSortingApp

2️⃣ Create a Virtual Environment (Recommended)

python -m venv venv
venv\Scripts\activate         # On Windows
# source venv/bin/activate   # On macOS/Linux

3️⃣ Install the Required Packages

pip install -r requirements.txt

4️⃣ Start the Flask App

python app.py

Then, open your browser and visit:
👉 http://127.0.0.1:5000


---


🗂 Stages of the Project

1. Data Collection and Preprocessing


2. Building and Training CNN Model


3. Evaluation and Accuracy Tuning


4. Saving Model and Class Indices


5. Flask Web App Development


6. Integration and Deployment




---

📊 Dataset

Total Images: 32,769

Total Classes: 16 categories (8 fresh + 8 rotten types)

Stored in:

SmartSortingApp/dataset/train

SmartSortingApp/dataset/test



> Class indices:
{'freshapples': 0, 'freshbanana': 1, ..., 'rottentomato': 15}




---

🧰 Technologies Used

Python

TensorFlow / Keras

Flask

HTML + CSS

Google Colab

---

🧪 Model Performance

Accuracy: ~95%

Loss: Very Low (improved via tuning and correct indexing)

Used Transfer Learning and custom CNN experiments



---

📄 Context

Food industries lose significant revenue due to improper classification of spoiled produce. This project solves the problem by:

Automating classification through image input.

Assisting in smart packaging and quality control.

Supporting farmers and vendors by detecting rotten produce before sale


📂 Project Structure
SmartSortingApp/
│
├── dataset/
│   ├── train/
│   └── test/
├── model/
│   └── fruit_classifier.h5
│   └── class_indices.json
├── app.py
├── templates/
    └── index.html
├── static/
│   └── style.css
├── README.md
└── requirements.txt
----
