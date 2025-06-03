import sys
import json
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

#  Updated Skill-to-Role Mapping (Deep Industry-Level Analysis)
skill_to_role_mapping = {
    # Software Engineering
    'python': ['Software Engineer', 'Backend Developer', 'Data Scientist', 'ML Engineer'],
    'java': ['Software Engineer', 'Backend Developer'],
    'c++': ['Software Engineer', 'Game Developer'],
    'javascript': ['Software Engineer', 'Frontend Developer', 'Full Stack Developer'],
    'typescript': ['Frontend Developer', 'Full Stack Developer'],
    'react': ['Frontend Developer', 'Full Stack Developer'],
    'angular': ['Frontend Developer'],
    'vue': ['Frontend Developer'],
    'node': ['Backend Developer', 'Full Stack Developer'],
    'express': ['Backend Developer'],
    'django': ['Backend Developer', 'Full Stack Developer'],
    'flask': ['Backend Developer'],
    'spring': ['Backend Developer'],

    # Data Science & AI
    'machine learning': ['Data Scientist', 'ML Engineer'],
    'deep learning': ['ML Engineer', 'AI Researcher'],
    'tensorflow': ['ML Engineer', 'AI Researcher'],
    'pytorch': ['ML Engineer', 'AI Researcher'],
    'nlp': ['ML Engineer', 'AI Researcher'],
    'computer vision': ['ML Engineer', 'AI Researcher'],
    'data analysis': ['Data Analyst', 'Data Scientist'],
    'big data': ['Data Engineer', 'Data Scientist'],
    'sql': ['Data Analyst', 'Data Engineer', 'Database Administrator'],
    'mongodb': ['Database Administrator', 'Backend Developer'],
    'hadoop': ['Data Engineer'],

    # DevOps & Cloud
    'aws': ['Cloud Engineer', 'DevOps Engineer'],
    'azure': ['Cloud Engineer', 'DevOps Engineer'],
    'gcp': ['Cloud Engineer'],
    'docker': ['DevOps Engineer'],
    'kubernetes': ['DevOps Engineer'],
    'ci/cd': ['DevOps Engineer'],
    'terraform': ['DevOps Engineer'],

    # Product & Design
    'product management': ['Product Manager'],
    'agile': ['Product Manager', 'Scrum Master'],
    'scrum': ['Scrum Master'],
    'figma': ['UX Designer'],
    'ui': ['UX Designer'],
    'ux': ['UX Designer'],
    'user research': ['UX Designer', 'Product Manager'],

    # Cybersecurity
    'penetration testing': ['Cybersecurity Engineer'],
    'network security': ['Cybersecurity Engineer'],
    'ethical hacking': ['Cybersecurity Engineer'],
    'encryption': ['Cybersecurity Engineer'],

    # Blockchain & Web3
    'solidity': ['Blockchain Developer'],
    'web3': ['Blockchain Developer'],
    'ethereum': ['Blockchain Developer'],

    # Soft Skills (Bonus Analysis)
    'leadership': ['Manager', 'Team Lead'],
    'communication': ['Manager', 'Team Lead', 'Product Manager'],
    'teamwork': ['Any Role']
}

#  Extract Text from Resume
def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        return text.strip()
    except Exception as e:
        print(json.dumps({"error": f"Error reading PDF: {str(e)}"}))
        sys.exit(1)

#  Extract Skills from Resume
def extract_skills(resume_text):
    resume_text_lower = resume_text.lower()
    found_skills = [skill for skill in skill_to_role_mapping if skill in resume_text_lower]
    return found_skills

#  Predict Role from Extracted Skills
def predict_roles(skills):
    matched_roles = set()
    for skill in skills:
        matched_roles.update(skill_to_role_mapping.get(skill, []))
    return list(matched_roles) if matched_roles else ["General Software Engineer"]  # Default role

#  Dummy ML Model for Role Prediction (Training)
data = {
    "resume": [
        "Python, Machine Learning, TensorFlow",
        "SQL, Data Analysis, Big Data",
        "Project Management, Agile, Scrum",
        "React, JavaScript, HTML, CSS",
        "Python,OOP,Data Structures and Algorithms,Git,SQL",
        "Javascript,React,Node.js,MongoDB,REST API",
        "Node.js,Express.js,SQL,Microservices,Docker",
        "Pyhton,Machine Learning,SQL,TensorFlow,Data Visualization"
    ],
    "role": [
        "ML Engineer",
        "Data Analyst",
        "Product Manager",
        "Frontend Developer",
        "Software Engineer",
        "Full Stack Developer",
        "Backend Developer",
        "Data Scientist"
    ]
}

df = pd.DataFrame(data)
vectorizer = TfidfVectorizer()
X_vectors = vectorizer.fit_transform(df["resume"])
model = LogisticRegression()
model.fit(X_vectors, df["role"])

#  Resume Scoring Function
def calculate_resume_score(skills):
    base_score = 0  # Start with a base score
    skill_boost = len(skills) * 5  # More skills → Higher score
    role_weight = sum(10 for skill in skills if skill in ["python", "sql", "machine learning", "aws", "docker"])  # Critical industry skills
    score = min(base_score + skill_boost + role_weight, 100)  # Cap at 100
    return score

#  Resume Analysis Function
def analyze_resume(pdf_path):
    resume_text = extract_text_from_pdf(pdf_path)
    if not resume_text:
        print(json.dumps({"error": "Empty resume text extracted"}))
        sys.exit(1)

    skills = extract_skills(resume_text)
    suggested_roles = predict_roles(skills)

    # ML Model Prediction
    resume_vector = vectorizer.transform([" ".join(skills)]) if skills else vectorizer.transform([resume_text])
    predicted_role = model.predict(resume_vector)[0]

    # Calculate Score
    resume_score = calculate_resume_score(skills)

    # Improvement Suggestions
    improvements = []
    if "python" not in skills and "java" not in skills:
        improvements.append("Consider adding Python or Java programming skills.")
    if "sql" not in skills and "mongodb" not in skills:
        improvements.append("Consider improving database knowledge (SQL, NoSQL).")
    if "cloud" not in skills and "aws" not in skills and "azure" not in skills:
        improvements.append("Cloud expertise (AWS, Azure) is highly valued in modern tech.")
    if "data structures" not in skills and "algorithms" not in skills:
        improvements.append("Strengthen knowledge of Data Structures and Algorithms for better problem-solving.")
    if "git" not in skills:
        improvements.append("Include Git and version control experience to improve collaboration.")
    if "javascript" in skills and "typescript" not in skills:
        improvements.append("Consider learning TypeScript for better scalability in frontend and backend development.")
    if "react" in skills and "redux" not in skills:
        improvements.append("Learn Redux for better state management in React applications.")
    if "node" in skills and "express" not in skills:
        improvements.append("Gain experience with Express.js for better backend development in Node.js.")
    if "sql" in skills and "mongodb" not in skills:
        improvements.append("Consider learning MongoDB for NoSQL database expertise.")
    if "microservices" not in skills:
        improvements.append("Gain knowledge of Microservices architecture for scalable backend systems.")
    if "python" in skills and "pandas" not in skills:
        improvements.append("Learn Pandas for efficient data manipulation in Python.")
    if "machine learning" in skills and "statistics" not in skills:
        improvements.append("Enhance understanding of Statistics for better ML model performance.")
    if "deep learning" in skills and "nlp" not in skills and "computer vision" not in skills:
        improvements.append("Consider specializing in NLP or Computer Vision for advanced AI applications.")

    #  Final Output
    result = {
        "suggestedRoles": list(set(suggested_roles + [predicted_role])),  # Combine ML + keyword roles
        "resumeScore": resume_score,
        "improvements": improvements
    }
    print(json.dumps(result))  #  JSON output for API

pdf_path = r"C:\Users\Minal\Desktop\SEM6\Resume_Analyzer\resume2_SE_3.pdf"

# Run the resume analysis
analyze_resume(pdf_path)

def match_resume_to_job(resume_text, jd_text):
    documents = [resume_text, jd_text]
    vectors = vectorizer.transform(documents)
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(similarity * 100, 2)

def rank_candidates(resume_paths, jd_text=None):
    results = []
    for path in resume_paths:
        resume_text = extract_text_from_pdf(path)
        skills = extract_skills(resume_text)
        score = calculate_resume_score(skills)
        match_score = match_resume_to_job(resume_text, jd_text) if jd_text else None
        results.append({
            "file": path,
            "score": score,
            "match": match_score
        })
    return sorted(results, key=lambda x: x["match"] or x["score"], reverse=True)

import streamlit as st
import pandas as pd
# import joblib

# model=joblib.load("resume_model.joblib")
st.title("Resume Analyzer")

uploaded_file = st.file_uploader("Upload a Resume PDF")
jd_input = st.text_area("Paste Job Description (Optional)")

if uploaded_file:
    with open(r"C:\Users\Minal\Desktop\SEM6\Resume_Analyzer\resume2_SE_3.pdf", "wb") as f:
        f.write(uploaded_file.read())
    text = extract_text_from_pdf(r"C:\Users\Minal\Desktop\SEM6\Resume_Analyzer\resume2_SE_3.pdf")
    skills = extract_skills(text)
    roles = predict_roles(skills)
    score = calculate_resume_score(skills)
    match = match_resume_to_job(text, jd_input) if jd_input else None

    st.subheader("Results")
    st.write("**Skills:**", skills)
    st.write("**Suggested Roles:**", roles)
    st.write("**Resume Score:**", score)
    if match:
        st.write("**Job Match %:**", match)
