smart-sorting-transfer-learning-for-identifying-rotten-fruits-and-vegetables

---

📘 Smart-Sorting – Fruit and Vegetable Freshness Detection 


---

🔬 Smart Sorting is a deep learning-based solution to classify fruits and vegetables as fresh or rotten using image recognition. It is designed to help automate sorting processes in the agricultural and food industries.



🧠 Objective

In this project, the aim is to:

Identify whether a fruit/vegetable is fresh or rotten.

Categorize input images into one of 26 classes using a trained CNN model.

Provide a web interface where users can upload an image and get the predicted class with confidence.




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

Heroku (for deployment)



---

🧪 Model Performance

Accuracy: ~95%

Loss: Very Low (improved via tuning and correct indexing)

Used Transfer Learning and custom CNN experiments



---

📂 Project Structure
SmartSortingApp/
│
├── dataset/
│   ├── train/
│   └── test/
│
├── model/
│   └── fruit_classifier.h5
│   └── class_indices.json
│
├── app.py
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── README.md
└── requirements.txt


---

📄 Context

Food industries lose significant revenue due to improper classification of spoiled produce. This project solves the problem by:

Automating classification through image input.

Assisting in smart packaging and quality control.

Supporting farmers and vendors by detecting rotten produce before sale

