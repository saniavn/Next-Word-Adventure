

import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import nltk
from nltk.util import ngrams
import os
import json
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import uuid
from flask import send_file, after_this_request



# Download necessary resources
nltk.download('punkt', quiet=True)

app = Flask(__name__)
# Define the directory to save user files
USER_DATA_DIR = 'user_data'

# Ensure the directory exists
if not os.path.exists(USER_DATA_DIR):
    os.makedirs(USER_DATA_DIR)

class NGramModel:
    """
    An n-gram model.
    """

    def __init__(self, train_text, n=2):
        self.n = n
        self.model = defaultdict(Counter)
        self.ngram_occurrences = defaultdict(lambda: defaultdict(int))
        self.create_ngram_model(train_text)



    def create_ngram_model(self, text):

        punctuation_to_keep = {'.', '!', '?'}
        words = [word for word in nltk.word_tokenize(text.lower()) if word.isalnum() or word in punctuation_to_keep]
        n_grams = list(ngrams(words, self.n))

        for gram in n_grams:
            self.model[gram[:-1]][gram[-1]] += 1

        for context in self.model:
            total = sum(self.model[context].values())
            for word in self.model[context]:
                self.ngram_occurrences[context][word] = self.model[context][word]
                self.model[context][word] /= total


    def predict_next_words(self, context):
        print("Current context for prediction:", context)
        if context in self.model:
            next_words = sorted(
                (word, prob) for word, prob in self.model[context].items() if word.isalnum()
            )
            print("Predicted words and probabilities:", next_words)
            return [
                {"word": word, "probability": prob, "count": self.ngram_occurrences[context][word]}
                for word, prob in next_words[:5]
            ]
        return []


@app.route('/')
def introduction():
    return render_template('introduction.html')

@app.route('/index', methods=['GET'])
def index():
    user_file = request.args.get('user_file', 'default_user.json')
    print("Received user file for ingredient interaction:", user_file)
    return render_template('index.html', user_file=user_file)

@app.route('/start_game', methods=['POST'])
def start_game():

    data = request.json
    user_name = data['userName']
    age_range = data['ageRange']
    selected_sentences = data.get('selectedSentences', [])
    student_answer = data.get('studentAnswer', '')
    user_id = str(uuid.uuid4())


    filename = f"{user_name}_{age_range}_{user_id}.json"
    filepath = os.path.join(USER_DATA_DIR, filename)

    user_data = {
        'userId': user_id,
        'userName': user_name,
        'ageRange': age_range,
        'selectedSentences': selected_sentences,
        'examples': [],
        'studentAnswer': student_answer,
        'step1_inputs': [],
        'step2_inputs': [],
        'step3_results': []
    }

    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            existing_data = json.load(file)
            existing_data['selectedSentences'] = list(
                set(existing_data.get('selectedSentences', []) + selected_sentences))
            existing_data['studentAnswer'] = student_answer
            user_data = existing_data

    with open(filepath, 'w') as file:
        json.dump(user_data, file, indent=4)

    return jsonify({'success': True, 'filename': filename})

@app.route('/save_example', methods=['POST'])
def save_example():
    data = request.json
    filename = data['filename']
    sentence = data['sentence']
    prediction = data['prediction']

    filepath = os.path.join(USER_DATA_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r+') as file:
            user_data = json.load(file)
            if 'examples' not in user_data:
                user_data['examples'] = []
            user_data['examples'].append({
                'sentence': sentence,
                'prediction': prediction
            })
            if 'selectedSentences' not in user_data:
                user_data['selectedSentences'] = []
            if sentence not in user_data['selectedSentences']:
                user_data['selectedSentences'].append(sentence)
            file.seek(0)
            json.dump(user_data, file, indent=4)
            file.truncate()
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': 'File not found'})

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    user_file = data.get('user_file')
    sentences = data.get('sentences', [])
    n = int(data.get('n', 2))
    input_sentence = data.get('input_sentence', '')

    if len(sentences) == 0:
        return jsonify({
            "error": "Oops! We need some sentences to learn from first. Please type a few in Step 2!",
            "explanation": "Without training data, the model can't guess the next word. Let's get started!"
        }), 400

    text = ' '.join(sentences)
    ngram_model = NGramModel(text, n)

    words = nltk.word_tokenize(input_sentence.lower())
    if len(words) < n - 1:
        return jsonify({"error": f"Your sentence needs at least {n - 1} words for our guessing game!"}), 400

    context = tuple(words[-(n - 1):])
    predictions = ngram_model.predict_next_words(context)

    if user_file:
        filepath = os.path.join(USER_DATA_DIR, user_file)
        if os.path.exists(filepath):
            with open(filepath, 'r+') as file:
                user_data = json.load(file)
                if 'step1_inputs' not in user_data:
                    user_data['step1_inputs'] = []
                user_data['step1_inputs'].append(input_sentence)
                if 'step2_inputs' not in user_data:
                    user_data['step2_inputs'] = []
                user_data['step2_inputs'].extend(sentences)
                if 'step3_results' not in user_data:
                    user_data['step3_results'] = []
                user_data['step3_results'].append({
                    'input_sentence': input_sentence,
                    'n': n,
                    'predictions': predictions
                })
                file.seek(0)
                json.dump(user_data, file, indent=4)
                file.truncate()

    return jsonify({
        "predictions": predictions,
        "explanation": f"We used something called {n}-grams to see what word might come next after '{' '.join(context)}'.",
        "highlight": ' '.join(context)
    })





## download and delete user data
@app.route('/download_and_delete', methods=['POST'])
def download_and_delete():
    data = request.get_json()
    user_file = data['userFile']
    file_path = os.path.join(USER_DATA_DIR, user_file)

    def remove_file():
        try:
            os.remove(file_path)
            print(f"Successfully deleted {file_path}")
        except Exception as error:
            print(f"Error when trying to delete file {file_path}: {error}")


    try:
        response = send_file(file_path, as_attachment=True)

        @after_this_request
        def cleanup(response):
            remove_file()
            return response
    except Exception as e:
        print(f"Failed to send or delete file: {str(e)}")
        return jsonify({'success': False, 'message': 'Failed to send or delete file.'}), 500

    return response



@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.get_json()
    user_file = data.get('userFile')
    general_feedback = data.get('generalFeedback')  # Renamed for clarity (software question )
    model_thinking_feedback = data.get('modelThinkingFeedback')  # New (software question )

    # Check for missing data
    if not user_file or not general_feedback or not model_thinking_feedback:
        return jsonify({'success': False, 'error': 'Missing user file or feedback details'}), 400

    filepath = os.path.join(USER_DATA_DIR, user_file)
    if os.path.exists(filepath):
        with open(filepath, 'r+') as file:
            user_data = json.load(file)
            # Append or create feedback list for general feedback (software question)
            if 'generalFeedbacks' not in user_data:
                user_data['generalFeedbacks'] = []
            user_data['generalFeedbacks'].append(general_feedback)

            # Append or create feedback list for model thinking feedback (software question)
            if 'modelThinkingFeedbacks' not in user_data:
                user_data['modelThinkingFeedbacks'] = []
            user_data['modelThinkingFeedbacks'].append(model_thinking_feedback)

            file.seek(0)
            json.dump(user_data, file, indent=4)
            file.truncate()
    else:
        return jsonify({'success': False, 'error': 'User file does not exist'}), 404

    return jsonify({'success': True, 'message': 'Feedback submitted successfully'})



if __name__ == '__main__':
    app.run(debug=True, port=5003)
