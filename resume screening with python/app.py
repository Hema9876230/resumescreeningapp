import streamlit as st
import pickle
import re
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load pre-trained models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Function to clean resume text
def cleanResume(resume_text):
    # Remove URLs (http links)
    cleanTxt = re.sub(r'http\S+', ' ', resume_text)

    # Remove occurrences of 'RT' and 'cc'
    cleanTxt = re.sub(r'RT|cc', ' ', cleanTxt)

    # Remove mentions (words starting with @)
    cleanTxt = re.sub(r'@\S+', ' ', cleanTxt)

    # Remove special characters
    cleanTxt = re.sub(r'[^a-zA-Z0-9\s]', ' ', cleanTxt)

    # Remove hashtags (words starting with #)
    cleanTxt = re.sub(r'#\S+', ' ', cleanTxt)

    # Remove non-ASCII characters
    cleanTxt = re.sub(r'[^\x00-\x7f]', ' ', cleanTxt)

    # Remove extra whitespace
    cleanTxt = re.sub(r'\s+', ' ', cleanTxt).strip()

    return cleanTxt

# Custom HTML and CSS for styling
def custom_css():
    st.markdown(
        """
        <style>
        body {
             font-family: 'Arial', sans-serif;
            background-color: #000000;  /* Black background */
            color : #FFFFFF;  /* White text */
            margin: 0;
            padding: 0;
        }
        .header {
            background-color: #4CAF50;
            padding: 20px;
            text-align: center;
            color: white;
            font-size: 30px;
            font-weight: bold;
            border-radius: 5px;
        }
        .upload-area {
            margin: 20px auto;
            text-align: center;
            padding: 10px;
            border: 2px dashed #4CAF50;
            border-radius: 10px;
            background-color: #fff;
        }
        .result {
            background-color: #4CAF50;
            color: white;
            padding: 20px;
            margin-top: 20px;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
            text-align: center;
        }
        .footer {
            margin-top: 20px;
            text-align: center;
            font-size: 14px;
            color: #888;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Streamlit app
def main():
    # Apply custom CSS
    custom_css()

    # Header
    st.markdown('<div class="header">Resume Screening App</div>', unsafe_allow_html=True)

    # Category mapping for prediction
    category_mapping = {
        15: "Java Developer",
        23: "Testing",
        8: "DevOps Engineer",
        20: "Python Developer",
        24: "Web Designing",
        12: "HR",
        13: "Hadoop",
        3: "Blockchain",
        10: "ETL Developer",
        18: "Operations Manager",
        6: "Data Science",
        22: "Sales",
        16: "Mechanical Engineer",
        1: "Arts",
        7: "Database",
        11: "Electrical Engineer",
        14: "Health and Fitness",
        9: "DotNet Developer",
        2: "Automation Testing",
        17: "Network Security Manager",
        21: "SAP Developer",
        5: "Civil Engineer",
        0: "Advocate",
    }

    # File upload section
    st.markdown('<div class="upload-area">Upload your resume (PDF or TXT)</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["pdf", "txt"])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')  # Try UTF-8 decoding
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')  # Fallback to Latin-1 decoding

        # Clean the uploaded resume
        cleaned_resume = cleanResume(resume_text)

        # Extract features using the TF-IDF vectorizer
        resume_features = tfidf.transform([cleaned_resume])

        # Predict using the classifier
        prediction_id = clf.predict(resume_features)[0]

        # Map prediction ID to category name
        category_name = category_mapping.get(prediction_id, "Unknown")

        # Display results
        st.markdown(f'<div class="result">Predicted Job Category: {category_name}</div>', unsafe_allow_html=True)

    # Footer
    st.markdown('<div class="footer">&copy; 2024 Resume Screening App | Built with Streamlit</div>', unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
