from flask import Flask, render_template, request, jsonify
import json
import fitz
import spacy
from transformers import pipeline
import re

app = Flask(__name__)

# Constants declaration
PDF_PATH = 'java.pdf'
CHUNK_SIZE = 700

# Custom Exceptions
class PDFExtractionError(Exception):
    pass

class BotResponseError(Exception):
    pass

# Load data from a JSON file
def load_json_data(file_path='data.json'):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError("JSON file not found. Please make sure the file exists.")

# Text conversion from pdf
def extract_text_from_pdf(pdf_path):
    try:
        with fitz.open(pdf_path) as doc:
            return " ".join(page.get_text() for page in doc)
    except Exception as e:
        raise PDFExtractionError("Error extracting text from PDF. Please check the PDF file.") from e

# Relevant text
def search_pdf(query, pdf_text):
    return pdf_text.lower().find(query.lower()) != -1

# Function to get PDF answer using BERT/ Pipeline
def get_pdf_answer_bert(query, pdf_text, max_length=200):
    relevant_text = ""
    chunks = [pdf_text[i:i + max_length] for i in range(0, len(pdf_text), max_length)]
    for chunk in chunks:
        if query.lower() in chunk.lower():
            relevant_text += chunk

    if relevant_text:
        try:
            summarizer = pipeline("summarization")
            summary = summarizer(relevant_text, max_length=500, min_length=50, length_penalty=1.0, num_beams=2)
            return summary[0]['summary_text']
        except Exception as e:
            raise BotResponseError("Error generating summary using BERT pipeline.") from e
    else:
        return None

# User input analysis
def analyze_user_input(user_input, nlp_model):
    try:
        doc = nlp_model(user_input)
        entities = [ent.text for ent in doc.ents]
        tokens = [token.text for token in doc]
        return entities, tokens
    except Exception as e:
        raise Exception("Error analyzing user input.") from e

# Bot response
def get_bot_response(user_input, entities, tokens, json_data):
    try:
        for item in json_data['questions']:
            if any(entity.lower() in item['question'].lower() for entity in entities) or any(
                    token.lower() in item['question'].lower() for token in tokens
            ):
                return item['answer']

        pdf_text = extract_text_from_pdf(PDF_PATH)

        # Handling questions like "What is_____?"
        if user_input.lower().startswith("what is ") and user_input.endswith("?"):
            entity = user_input[8:-1].strip()
            if search_pdf(entity, pdf_text):
                return get_pdf_answer_bert(entity, pdf_text)
            else:
                return "Sorry, I don't have information about " + entity

        # Handling questions like "Who is ____?"
        if user_input.lower().startswith("who is ") and user_input.endswith("?"):
            who = user_input[7:-1].strip()
            if search_pdf(who, pdf_text):
                return get_pdf_answer_bert(who, pdf_text)
            else:
                return "Sorry, I don't have information about " + who

        # Handling enlisting type of question
        if user_input.lower().startswith("enlist ") or user_input.lower().startswith("list "):
            Enlist_words = user_input.split(" ", 1)[1]

            if search_pdf(Enlist_words, pdf_text):
                items = get_pdf_answer_bert("Enlist " + Enlist_words, pdf_text)
                if items:
                    formatted_items = "\n".join([f"- {item}" for item in items.split("\n")])
                    return f"Here is a list of {Enlist_words}:\n{formatted_items}"
                else:
                    return f"Sorry, I don't have information about {Enlist_words}"
            else:
                return f"Sorry, I don't have information about {Enlist_words}"

        # Handling "How to" questions
        if user_input.lower().startswith("how to ") and user_input.endswith("?"):
            process = user_input[7:-1].strip()
            if search_pdf(process, pdf_text):
                return get_pdf_answer_bert(process, pdf_text)
            else:
                return "Sorry, I don't have information about how to " + process

        pdf_answer = get_pdf_answer_bert(user_input, pdf_text)
        if pdf_answer:
            return pdf_answer

        return "Sorry, I didn't find information in the pdf."
    except (PDFExtractionError, BotResponseError) as e:
        return str(e)
    except Exception as e:
        return "An unexpected error occurred. Please try again later."

# Routes for flask
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    try:
        user_input = request.form['user_input']
        entities, tokens = analyze_user_input(user_input, spacy.load("en_core_web_sm"))
        response = get_bot_response(user_input, entities, tokens, load_json_data())
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
