from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
from tensorflow.keras.models import load_model
import csv
import time
from Bio import SeqIO
from Bio.SeqUtils import seq1
import os

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'


def preprocess_sequences(sequences):
    processed_sequences = []
    for seq in sequences:
        changes = seq.count('B') + seq.count('J') + seq.count('O') + seq.count('U') + seq.count('Z')
        seq = seq.replace('B', 'X').replace('J', 'X').replace('O', 'X').replace('U', 'X').replace('Z', 'X')
        formatted_sequence = ' '.join(seq)
        processed_sequences.append(formatted_sequence)
    print("Pre-processing done")
    return processed_sequences


def encode_sequences(sequences):
    tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
    model = TFAutoModel.from_pretrained("Rostlab/prot_bert_bfd")
    encoded_sequences = []
    for seq in sequences:
        inputs = tokenizer(seq, return_tensors="tf", padding=True, truncation=True, max_length=1000)
        outputs = model(**inputs)
        embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1)
        encoded_sequences.append(embeddings.numpy())
    print("Sequences encoded")
    return np.array(encoded_sequences)


def predict_classes(X, BERTmodel, mlb, thresh, processed_sequences, ec_dictionary):
    #print ("EC dictionary 3 key:", ec_dictionary[2])
    #print ("TYpe of key:", type(ec_dictionary.keys()))
    y_pred = BERTmodel.predict(X)
    y_pred_binary = (y_pred > thresh).astype(int)
    y_pred_probabilities_filtered = y_pred * y_pred_binary
    predicted_class = mlb.inverse_transform(y_pred_binary)
    result_data = []
    for index, elements in enumerate(predicted_class):
        filtered_probs = (y_pred_probabilities_filtered[index][y_pred_binary[index] == 1]) * 100
        
        #Replace spaces in sequences
         
        row_data = [index, processed_sequences[index].replace(' ', '')]
        row_data = [index, processed_sequences[index].replace(' ', '')]
        
        ##################ADDITION
        # Replace class names with values from ec_dictionary
        elements_with_values = [ec_dictionary.get(int(class_name), '') for class_name in elements]
        
        for class_value, prob in zip(elements_with_values, filtered_probs):
            row_data.extend([class_value, prob])
           
        row_data.extend([np.nan] * (2 * (mlb.classes_.shape[0] - len(elements))))
        result_data.append(row_data)
    column_names = ['Index', 'Sequence']
    
    for i in range(1, mlb.classes_.shape[0] + 1):
        column_names.extend([f'Class_{i}', f'Probability_{i}'])
    df_result = pd.DataFrame(result_data, columns=column_names)
    return df_result


def validate_sequence_length(sequences):
    for seq in sequences:
        if not (30 <= len(seq) <= 1000):
            return False
    return True
    
    
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET'])
def predict():
    return render_template('predict.html')


@app.route('/prediction', methods=['POST'])
def prediction():
    # Get user input
    input_file = request.files['input_file']
    uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_fasta.fasta')
    #input_file.save(uploaded_file_path)
    
    if request.files['input_file'].filename != '':
        input_file.save(uploaded_file_path)
    elif request.form['sequence'] != '':
        with open(uploaded_file_path, "w") as fo:
            fo.write(request.form['sequence'])
    else:
        return render_template('predict.html', errorMessage='Please upload .fasta file or use sample sequence')

    # file_content = input_file.read()
    thresh = float(request.form['thresh'])

    # Load sequences from input file
    sequences = []
    with open(uploaded_file_path, "r") as fasta_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequences.append(str(record.seq))
        
    #####################################################################
    # Check if the number of sequences exceeds 20
    if len(sequences) > 20:
    
        error_message = "The input contains an excess of sequences. The maximum allowable number of sequences is 20."
        return render_template(
            'predict.html',
            result_file='',
            df_result='',
            errorMessage=error_message,
            successMessage=''
        )
    
    # Validate sequence length
    if not validate_sequence_length(sequences):
        error_message2 = "The length of all sequences must be between 30 and 1000."
        
        return render_template(
            'predict.html',
            result_file='',
            df_result='',
            errorMessage=error_message2,
            successMessage=''
        )
    
    
    #######################################################################
    
    # Preprocess sequences
    processed_sequences = preprocess_sequences(sequences)

    # Encode sequences
    X = encode_sequences(processed_sequences)

    # Load required files
    labels_id = pd.read_csv('label_code.csv')
    labels = labels_id["id"].str.split(";", n=None, expand=False)
    mlb = MultiLabelBinarizer()
    labels_encoded = mlb.fit_transform(labels)

    classname_file = 'EC_labels.csv'
    ec_dictionary = {}
    with open(classname_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            id_value = row['id']
            ec_value = row['EC']
            ec_dictionary[int(id_value)] = ec_value

    BERTmodel = load_model('./model/multilabel_protein_bertmodel.h5')

    # Predict classes
    print("Predicting enzyme classes")
    df_result = predict_classes(X, BERTmodel, mlb, thresh, processed_sequences, ec_dictionary)
    
    # Select only three classes and their probabilities
    df_result_top3 = df_result[['Sequence','Class_1', 'Probability_1', 'Class_2', 'Probability_2', 'Class_3', 'Probability_3']]
    df_result_top3[['Probability_1', 'Probability_2', 'Probability_3']] = df_result_top3[['Probability_1', 'Probability_2', 'Probability_3']].round(2)
    df_result_top3 = df_result_top3.fillna('')

    # Save DataFrame to CSV
    result_file_name = 'result.csv'
    result_path = './static/' + result_file_name
    df_result_top3.to_csv(result_path, index=False)

    return render_template(
        'predict.html',
        result_file=result_file_name,
        df_result=df_result_top3.to_html(),
        errorMessage='',
        successMessage=''
    )

@app.route('/downloads', methods=['GET'])
def download():
    return render_template('download.html')


@app.route('/help', methods=['GET'])
def help():
    return render_template('help.html')


@app.route('/team', methods=['GET'])
def team():
    return render_template('team.html')


@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html')

@app.route('/error', methods=['POST'])
def error():
    return render_template('error.html')
    
if __name__ == '__main__':
    app.run(debug=False)