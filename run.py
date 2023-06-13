from flask import Flask, render_template, request
import spacy
import pandas as pd

app = Flask(__name__)

nlp = None
synth = "synthesized.csv"
extjobs = None

def load_model_and_data():
    global nlp, extjobs
    try:
        nlp = spacy.load('en_core_web_md')
    except OSError:
        from spacy.cli import download
        download('en_core_web_md')
        nlp = spacy.load('en_core_web_md')
    
    extjobs = pd.read_csv(synth, index_col=0)

# Load the model and data during the initialization phase
load_model_and_data()

from flask import Flask, request, jsonify

@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    prompt = data['prompt']
    top_indexes = get_top_sentence_indexes(prompt)
    top_rows = extjobs.iloc[top_indexes]
    rows_list = top_rows.to_dict(orient='records')

    return jsonify(rows_list)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/process', methods=['POST'])
# def process():
#     prompt = request.form['prompt']
#     top_indexes = get_top_sentence_indexes(prompt)
#     top_rows = extjobs.iloc[top_indexes]
#     rows_list = top_rows.to_dict(orient='records')

#     return render_template('result.html', rows=rows_list)

def get_top_sentence_indexes(prompt):
    prompt_words = nlp(prompt.lower())
    prompt_words = [token.text for token in prompt_words if not token.is_punct and not token.is_stop]

    all_sentences = []

    for column in extjobs.columns:
        sentences = extjobs[column].dropna().tolist()
        tokens = [nlp(str(sentence).lower()) for sentence in sentences]
        
        for index, doc in enumerate(nlp.pipe(tokens, disable=["parser", "tagger", "ner"], batch_size=1000)):
            similarities = [doc.similarity(nlp(word.lower())) for word in prompt_words]
            max_similarity = max(similarities)
            all_sentences.append((sentences[index], max_similarity, index))

    sorted_sentences = sorted(all_sentences, key=lambda x: x[1], reverse=True)
    top_10_indexes = [item[2] for item in sorted_sentences[:10]]

    return top_10_indexes

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
