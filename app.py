from flask import Flask, request, render_template
import pickle
import pdfplumber
import docx
import os
import re

# Load trained model and transformer
tf = pickle.load(open("tf.pkl", "rb"))
clf = pickle.load(open("clf.pkl", "rb"))

# Category Mapping
category_mapping = {
    15: "Java Developer", 23: "Testing", 8: "DevOps Engineer", 20: "Python Developer",
    24: "Web Designing", 12: "HR", 3: "Blockchain", 10: "ETL Developer",
    18: "Operations Manager", 6: "Data Science", 22: "Sales", 1: "Arts",
    7: "Database", 11: "Electrical Engineering", 13: "HR", 19: "PMO",
    4: "Business Analyst", 9: "DotNet Developer", 2: "Automation Testing",
    21: "SAP Developer", 16: "Machine Learning", 17: "Network Security Engineer",
    14: "Network Engineer", 5: "Civil Engineer", 0: "Advocate"
}

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            return " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return " ".join([para.text for para in doc.paragraphs])
    return ""

def clean_text(text):
    text = re.sub(r'http\S+', ' ', text)  # Remove URLs
    text = re.sub(r'\b(RT|cc)\b', ' ', text)  # Remove 'RT' and 'cc'
    text = re.sub(r'#\S+', ' ', text)  # Remove hashtags
    text = re.sub(r'@\S+', ' ', text)  # Remove mentions
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'resume' not in request.files:
        return "No file uploaded"
    
    file = request.files['resume']
    if file.filename == '':
        return "No selected file"
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    resume_text = extract_text(file_path)
    cleaned_text = clean_text(resume_text)
    
    input_features = tf.transform([cleaned_text])
    prediction = clf.predict(input_features)[0]
    category_name = category_mapping.get(prediction, "Unknown")
    
    return render_template('result.html', category=category_name)

if __name__ == '__main__':
    app.run(debug=True)
