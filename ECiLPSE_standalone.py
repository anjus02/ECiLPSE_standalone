# ECiLPSE standalone code can be used to predict enzyme classification for bulk seuqnces. It is GPU supported 
# Input : sequence.fasta file containing enzyme sequence in FASTA format
# Output: result.csv containing Sequence, Top three predicted Classes and Probability score

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


def preprocess_sequences(sequences):
    processed_sequences = []
    for seq in sequences:
        changes = seq.count('B') + seq.count('J') + seq.count('O') + seq.count('U') + seq.count('Z')
        seq = seq.replace('B', 'X').replace('J', 'X').replace('O', 'X').replace('U', 'X').replace('Z', 'X')
        formatted_sequence = ' '.join(seq.upper())
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
    y_pred = BERTmodel.predict(X)
    y_pred_binary = (y_pred > thresh).astype(int)
    y_pred_probabilities_filtered = y_pred * y_pred_binary
    predicted_class = mlb.inverse_transform(y_pred_binary)
    result_data = []
    for index, elements in enumerate(predicted_class):
        filtered_probs = (y_pred_probabilities_filtered[index][y_pred_binary[index] == 1]) * 100
        
        row_data = [index, processed_sequences[index].replace(' ', '')]
        row_data = [index, processed_sequences[index].replace(' ', '')]
        
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


def process_input(input_file_path, thresh):
    if input_file_path:
        if not input_file_path.endswith('.fasta'):
            return None, "Please upload a file with .fasta extension"
            
    if input_file_path:
        sequences = []
        with open(input_file_path, "r") as fasta_file:
            for record in SeqIO.parse(fasta_file, "fasta"):
                sequences.append(str(record.seq))

        
        if not validate_sequence_length(sequences):
            return None, "The length of all sequences must be between 30 and 1000."

        processed_sequences = preprocess_sequences(sequences)
        X = encode_sequences(processed_sequences)
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

        df_result = predict_classes(X, BERTmodel, mlb, thresh, processed_sequences, ec_dictionary)
        
        df_result_top3 = df_result[['Sequence','Class_1', 'Probability_1', 'Class_2', 'Probability_2', 'Class_3', 'Probability_3']]
        df_result_top3[['Probability_1', 'Probability_2', 'Probability_3']] = df_result_top3[['Probability_1', 'Probability_2', 'Probability_3']].round(2)
        df_result_top3 = df_result_top3.fillna('')
        
        result_file_name = 'result.csv'
        df_result_top3.to_csv(result_file_name, index=False)

        return df_result_top3, None
    else:
        return None, "Please upload .fasta file or use sample sequence"
    

# Example usage
input_file_path = input('Enter the path of input file (sequence.fasta): ')  # Replace with actual path
thresh = float(input("Enter the threshold value (0.9, 0.8, 0.7): "))  # Replace with actual threshold value
result, error_message = process_input(input_file_path, thresh)
if error_message:
    print("Error:", error_message)
else:
    print("Result:")
    print(result)
