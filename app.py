from flask import Flask, render_template, request, jsonify
import json
import fitz
import spacy
from transformers import pipeline

app = Flask(__name__)

# Constants declaration

PDF_PATH = 'java.pdf'  
CHUNK_SIZE = 1000

# Load data from a JSON file
def load_json_data(file_path='data.json'):
    with open(file_path, 'r') as file:
        return json.load(file)

# text conversion from pdf
def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        return " ".join(page.get_text() for page in doc)

# relevant text
def search_pdf(query, pdf_text):
    return pdf_text.lower().find(query.lower()) != -1

# Function to get PDF answer using BERT/ Pipeline
def get_pdf_answer_bert(query, pdf_text, max_length=150):
    relevant_text = ""
    chunks = [pdf_text[i:i + max_length] for i in range(0, len(pdf_text), max_length)]
    for chunk in chunks:
        if query.lower() in chunk.lower():
            relevant_text += chunk

    if relevant_text:
        summarizer = pipeline("summarization")
        summary = summarizer(relevant_text, max_length=200, min_length=50, length_penalty=1.5, num_beams=4)
        return summary[0]['summary_text']
    else:
        return None

# User input analysis
def analyze_user_input(user_input, nlp_model):
    doc = nlp_model(user_input)
    entities = [ent.text for ent in doc.ents]
    tokens = [token.text for token in doc]
    return entities, tokens

# bot response
def get_bot_response(user_input, entities, tokens, json_data):
    for item in json_data['questions']:
        if any(entity.lower() in item['question'].lower() for entity in entities) or any(
                token.lower() in item['question'].lower() for token in tokens
        ):
            return item['answer']

    pdf_text = extract_text_from_pdf(PDF_PATH)
    
    #handling questions like what in the pdf
    if user_input.lower().startswith("what is") and user_input.endswith("?"):
        entity = user_input[8:-1].strip()
        if search_pdf(entity, pdf_text):
            return get_pdf_answer_bert(entity,pdf_text)
        else: 
            return "Sorry I don't have information about " + entity   
             
    #handling Questions like Who in the pdf
    if user_input.lower().startswith("who is") and user_input.endswith("?"):
        entity = user_input[6:-1].strip()
        if search_pdf (entity,pdf_text):
            return get_pdf_answer_bert(entity, pdf_text)
        else:
            return "Sorry i did not found relevant information"
    
    
    pdf_answer = get_pdf_answer_bert(user_input, pdf_text)
    if pdf_answer:
        return pdf_answer

    return "Sorry, I didn't found information in pdf."

# Routes for flask 
@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    entities, tokens = analyze_user_input(user_input, spacy.load("en_core_web_sm"))
    response = get_bot_response(user_input, entities, tokens, load_json_data())
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
