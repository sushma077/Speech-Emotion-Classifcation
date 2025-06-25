# Speech-Emotion-Classifcation

## Project Overview
This project focuses on building a machine learning system that classifies emotions from speech audio clips using the RAVDESS dataset (audio-only variant). It leverages Librosa for audio feature extraction and uses a Multi-Layer Perceptron (MLP) as the classification model. The system is designed to identify emotional states based on speech signals by analyzing Mel Frequency Cepstral Coefficients (MFCCs).

The emotion categories used in this classification task include:

  •	Neutral
  
  •	Calm
  
  •	Happy
  
  •	Sad
  
  •	Angry
  
  •	Fearful
  
  •	Disgust
  
  •	Surprised
  
The goal is to accurately detect the emotional tone of speech samples using supervised learning techniques on extracted audio features.

________________________________________
## Data Preprocessing Pipeline

Dataset: RAVDESS Audio Files
Each audio file in the RAVDESS dataset follows a naming convention like:
03-01-01-01-01-01-03.wav, where the components indicate:

  •	Modality: 03 → Audio only
  
  •	Vocal Channel: 01 → Speech
  
  •	Emotion: Integer values (01–08)
  
  •	Other parts (Intensity, Statement, Repetition, Actor): used for organizing the dataset

## Processing Steps

1.	Audio Loading
   
     o	Audio samples are loaded using librosa.load, ensuring the original sample rate is preserved.


3.	Label Extraction
   
     o	Emotion labels are parsed from the filenames using string manipulation techniques.
  
     o	These numeric emotion codes are mapped to their corresponding human-readable names.

5.	Feature Extraction
   
     o	MFCC features are extracted using librosa.feature.mfcc.
  
     o	The mean of MFCC coefficients over time is computed for each audio file and used as the feature vector.

7.	Dataset Creation
   
     o	The extracted MFCC features are stored in X and the corresponding labels in y.
  
     o	The dataset is split into training and testing subsets using train_test_split.

________________________________________
## Modeling Approach

Classifier: Multi-Layer Perceptron (MLP)
The classification task is performed using the MLPClassifier from sklearn.neural_network.

  •	Model Configuration
  
    o	Hidden Layers: 2
    
    o	Activation Function: ReLU
    
    o	Solver: Adam
  
  •	Training
  
    o	The model is trained on the MFCC feature vectors.
    
    o	Target labels are encoded using LabelEncoder to be compatible with the MLP model.

________________________________________
## End-to-End Workflow

  •	Read and load all audio file paths from the dataset
  
  •	Extract MFCC features from each sample
  
  •	Label the dataset based on file names
  
  •	Preprocess the dataset (feature scaling and label encoding)
  
  •	Train the MLP model on training data
  
  •	Evaluate model performance on test data

________________________________________
## Model Evaluation & Saving

  •	The trained MLP classifier is evaluated using accuracy metrics on the test set.
  
  •	The final trained model is saved as a .pth file (best_emotion_model.pth) using pickle, allowing for future inference on unseen audio samples.

## Accuracy Metrics

Before Reducing Features:

  •	Overall Accuracy: 77%
  
  •	Overall F1 Score:73.4%
  
  •	Accuracy of classes:

    1.	Neutral:92%
      
    3.	Calm: 76%
       
    5.	Happy: 62%
       
    7.	Sad: 75%
       
    9.	Angry: 76%
        
    11.	Fearful:82%
        
    13.	Disgust:70%
        
    15.	Surprised:76%
    
After Reducing Features:

  •	Overall Accuracy: 82%
  
  •	Overall F1 Score:81.7%
  
  •	Accuracy of classes:

    1.	Calm: 82%
       
    3.	Happy: 85%
       
    5.	Sad: 78%
       
    7.	Angry: 82%
       
    9.	Fearful:85%
        
    11.	Disgust:81%
      



