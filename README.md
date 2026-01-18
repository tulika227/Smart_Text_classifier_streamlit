# Smart_Text_classifier_streamlit
Smart Text Classifier using Machine Learning & Streamlit
Overview
The Smart Text Classifier is an end-to-end Machine Learning project that classifies business-related text queries into predefined categories such as Technical, Billing, and General.
The project uses Natural Language Processing (NLP) techniques for text representation, a supervised ML model for classification, and Streamlit to provide an interactive web interface.
This project simulates a real-world customer support automation system where incoming queries can be automatically routed to the appropriate department.

Problem Statement
Businesses receive a high volume of customer queries every day through emails, chats, and support tickets.
Manually categorizing these queries is:
Time-consuming
Error-prone
Not scalable

Solution
Build an automated text classification system that:
Accepts raw text queries
Predicts the appropriate business category
Displays results through a simple web application

Project Structure
Smart_Text_Classifier_Streamlit/
│
├── business_text_dataset.csv   # Labeled dataset
├── train_model.py              # Model training & evaluation
├── model.pkl                   # Saved trained model
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation

Machine Learning Methodology
🔹 Text Preprocessing
Conversion of text to lowercase
Removal of stopwords
Vectorization using TF-IDF (Term Frequency–Inverse Document Frequency)

🔹 Model Selection
Logistic Regression
Chosen because:
It performs well for text classification
It is fast and interpretable
It serves as a strong baseline model

ML Pipeline
Raw Text → TF-IDF Vectorizer → Logistic Regression → Category Prediction
