# resumescreeningapp
The Resume Screening App is a machine learning-powered web application designed to help recruiters and hiring managers automatically categorize resumes into relevant job roles. 
The Resume Screening App is a web application built with Streamlit that helps recruiters and hiring managers efficiently screen resumes. By leveraging machine learning and natural language processing (NLP) techniques, this app analyzes resumes, categorizes them into job roles, and ranks candidates based on their qualifications, making the recruitment process faster and more accurate.

The app utilizes a pre-trained classifier and TF-IDF vectorizer to classify resumes into predefined job categories such as Java Developer, Python Developer, HR, Data Science, and many more.

Features
Resume Upload: Users can upload resumes in PDF or TXT format.
Resume Text Extraction: Automatically extracts and processes the text from uploaded resumes.
Text Cleaning: The app cleans the resume text by removing URLs, mentions, hashtags, special characters, and extra spaces to ensure the classifier's effectiveness.
Job Category Prediction: Based on the resume content, the app predicts the most suitable job category for the candidate.
Custom Styling: The app has a user-friendly interface with custom styling for a clean, professional look.
Real-Time Results: Once a resume is uploaded, the job category prediction is displayed instantly.
Technologies Used
Streamlit: For building the web application interface.
Pickle: For loading pre-trained machine learning models (Classifier and TF-IDF vectorizer).
NLTK: For natural language processing tasks such as tokenization and stopword removal.
Regex: For cleaning and preprocessing the text (e.g., removing URLs, mentions, and special characters).
Python 3.x: For backend logic and processing.
