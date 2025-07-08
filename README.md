_Project Description_

**cnn_mnist.ipynb**
This project builds a Convolutional Neural Network (CNN) to classify handwritten digits using the MNIST dataset. It demonstrates the full image classification pipeline, including data preprocessing, model architecture, training, and evaluation.
Project Workflow

1. Load and explore the MNIST dataset (60,000 training images and 10,000 test images of digits 0–9).
2. Flatten/reshape input data if needed, or keep it in 2D format for CNN.
3. Normalize pixel values (0–255) by dividing by 255 to bring them into range [0, 1].
4. One-hot encode the output labels using to_categorical.
5. Build the CNN model using Keras Sequential API:
Convolutional layers (Conv2D)
MaxPooling layers
Dropout (for regularization)
Dense (fully connected) layers
6. Compile and train the model using categorical crossentropy loss.
7. Evaluate the model’s accuracy on the test set and make predictions.

 **nltk_sentiment_analysis.ipynb**
 This project performs sentiment analysis on Hindi text by following these key steps:

1. Loads Hindi sentences from a text file (SampleHindiText.txt).
2. Translates each sentence from Hindi to English using the GoogleTranslator from the deep_translator library.
3. Analyzes the translated English text using the VADER SentimentIntensityAnalyzer from the vaderSentiment package.
4. Prints the translated sentence along with a dictionary of sentiment scores (compound, neg, neu, pos).
5. Classifies each sentence as:
Positive (compound score ≥ 0.05)
Negative (compound score ≤ -0.05)
Neutral (otherwise)

**rnn_clothing.ipynb**
A Recurrent Neural Network (RNN) model trained on fashion-related data (e.g. custom clothing dataset).

1. Processes raw text data (e.g., product reviews in clothing domain).
2. Performs NLP preprocessing:
3. Removal of special characters
4. Lowercasing, tokenization, stopword removal
5. Lemmatization using WordNetLemmatizer
6. Uses TensorFlow/Keras to:
Tokenize and pad sequences
Build and train a neural network (RNN/LSTM likely in later cells).

**rnn_master_card.ipynb**
This project involves time series data preprocessing, modeling, and evaluation using deep learning techniques in Python. The script begins by importing essential libraries for data handling, visualization, preprocessing, modeling, and evaluation.

TensorFlow/Keras:
Sequential: To build a sequential neural network model.
Layers such as Dense, LSTM, Dropout, GRU, and Bidirectional are used to build deep learning models tailored for time series forecasting or sequence modeling.
SGD: A stochastic gradient descent optimizer.
set_seed: For setting reproducible results using a random seed.

Reproducibility
A fixed random seed is set using set_seed and np.random.seed() to ensure consistent results across different runs.

**tesseract_grayscale_conversion.ipynb**
A basic image processing project that uses Tesseract OCR to extract text from grayscale images. Includes image-to-text conversion techniques and grayscale preprocessing steps.
