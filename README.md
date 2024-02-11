# ECiLPSE

Enzymes, the molecular catalysts of life, play a pivotal role in various biological processes. With seven main classes—oxidoreductases, transferases, hydrolases, lyases, isomerases, ligases, and translocases—accurate enzyme classification is crucial for understanding their diverse functions. 

**ECiLPSE** is a novel enzyme classification prediction tool that leverages Prot-BERT encoding scheme and a bidirectional LSTM model.  Trained on a diverse dataset comprising 256,222 enzyme sequences, the model achieves granularity by predicting up to the fourth level of enzyme classification and classifying enzymes into 1991 distinct classes with an impressive prediction accuracy of 98%, making it a reliable choice for researchers seeking precision in enzyme analysis. 

## Contents

The files contained in this repository are as follows:
 * ``ECiLPSE_standalone.py``: main script to run predictions
 * ``sequence.fasta``: user input sequences (in FASTA format)
 * ``/model/multilabel_protein_bertmodel.h5``: folder containing ECiLPSE model
 * ``EC_labels.csv``: Enzyme classes
 * ``label_code.csv``: Numeric codes of enzyme classes

## Pre-requisite

* Python 3.11.7
* Numpy	1.26.3
* Pandas 2.1.4
* Keras 2.15.0
* Tensorflow 2.15.0
* Transformers 4.36.2
* Biopython 1.83
* Flask 3.0.0

## Usage

In order to run enzyme classification predictions with **ECiLPSE**, save input sequences in FASTA format in a single 
file (e.g. ```sequence.fasta```). Remember to give a name or ID to each structure.
 
1. Download this repository and ensure that all the files are present in the same folder when running the script.
2. Install pre-requisite libraries using ``pip install name_of_library``
2. Run ``ECiLPSE_standalone.py``. 
  ```bash
  python ECiLPSE_standalone.py 
  ```
   The script runs in the current directory.
   The script will prompt user to enter the path of ```sequence.fasta``` file, followed by the ```threshold``` value.
   The output is provided in a comma-separated (result.csv) file format, presenting predictions for one or more (top 3 in the case of multiple classes) main class, subclass, sub-subclass, or substrate class of ECs, along with associated probability scores
  
> **_NOTE:_** Remember to activate the corresponding conda environment before running the script, if applicable.

## Citation

If you use **ECiLPSE** in your publication, consider citing the [paper](https://--):
```
@ARTICLE{10.3389/fnins.2022.858126,
AUTHOR={Sharma, Anju and Kumar, Rajnish and Diwaker, Vineet and Garg, Prabha},   
TITLE={ECiLPSE: Enzyme classification integrating LSTM and Prot-BERT sequence encoding},      
JOURNAL={ },      
VOLUME={ },           
YEAR={},     
URL={https://},       
DOI={},      	
ISSN={}
}
```
