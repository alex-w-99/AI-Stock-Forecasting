from typing import Dict, List
import datetime, holidays
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from textblob import TextBlob
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

ONE_DAY = datetime.timedelta(days=1)
HOLIDAYS_US = holidays.US()

FINBERT_PRETRAINED_MODEL = "ProsusAI/finbert"
SENTIMENT_TOKENIZER = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path = FINBERT_PRETRAINED_MODEL
)
SENTIMENT_MODEL = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path = FINBERT_PRETRAINED_MODEL
)

def get_next_business_day(
    formatted_start_dtm: str
) -> datetime.datetime:
    """
    Purpose: Calculate the next business day in the US (excluding weekends and 
        specified holidays) following a given date.
    :param date_str: A string representing the date in the format 'YYYY-MM-DD'.
    :return: A datetime.datetime object representing the next business day.
    """
    dtm_dtm = datetime.datetime.strptime(formatted_start_dtm, '%Y-%m-%d')
    next_day = dtm_dtm + ONE_DAY
    while next_day.weekday() in holidays.WEEKEND or next_day in HOLIDAYS_US:
        next_day += ONE_DAY
    return next_day

def _get_finbert_sentiment_analysis(
    doc: List[str]
) -> np.ndarray:
    """
    Purpose: Analyze sentiment scores for a list of text documents using the
        FinBERT pre-trained sentiment analysis model.
    :param doc: List[str] representing a collection of text documents.
    :return: np.ndarray representing sentiment scores for each document in 
        the input list.
    """
    # generate sentiment scores for each doc using the pre-trained model
    pt_input_encodings = SENTIMENT_TOKENIZER(
        doc,
        padding = True,
        truncation = True,
        max_length = 512,
        return_tensors = "pt"  # PyTorch
    )
    model_outputs = SENTIMENT_MODEL(**pt_input_encodings)
    pt_predicted_probabilities = F.softmax(
        model_outputs.logits, 
        dim = -1
    )
    sentiment_scores = pt_predicted_probabilities.detach().cpu().numpy()
    # each row in the sentiment_scores array corresponds to a document;
    # each column in the array represents sentiment class probabilities
    return sentiment_scores

def _get_textblob_sentiment_analysis(
    doc = List[str]
) -> np.ndarray:
    """
    Purpose: Uses TextBlob to run NLP on a collection of documents. TextBlob 
        then determines a "polarity" (sentiment positivity/negativitiy) and 
        "subjectivity" (document objectivity/subjectivity) score for each.
    :param doc: List[str] representing a collection of text documents.
    :return textblob_sentiment_scores: np.ndarray representing the polarity-
        subjectivty rating pairs for each document. 
    """
    textblob_sentiment_scores_list = list()
    for sentence in doc:
        analysis = TextBlob(sentence)
        textblob_sentiment_scores = {
            "polarity": analysis.sentiment.polarity,  # closer to -1 means negative, closer to 1 means positive
            "subjectivity": analysis.sentiment.subjectivity  # 0 means totally objective; 1 means totally subjective
        }
        textblob_sentiment_scores_list.append(textblob_sentiment_scores)
    textblob_sentiment_scores = np.array([list(score.values()) for score in textblob_sentiment_scores_list])
    return textblob_sentiment_scores

def _get_sentiment_analysis(
    doc: List[str]
) -> np.ndarray:
    """
    Purpose: Determines and returns the sentiment (pos, neg, and neu) of a
        collection of documents. These figures are calculated by taking
        FinBERT's 3 sentiment percentage categories, and then adjusting them
        based on TextBlob's analysis of the text's polarity and subjectivity.
    :param doc: List[str] representing a collection of text documents.
    :return: np.ndarray representing the 3 sentiment percentages for
        each document in doc. 
    """
    finbert_sentiment_analyses =  _get_finbert_sentiment_analysis(doc = doc)
    textblob_sentiment_analyses = _get_textblob_sentiment_analysis(doc = doc)
    
    sentiment_scores = list()
    for i in range(len(doc)):
        finbert_analysis = finbert_sentiment_analyses[i]
        textblob_analysis = textblob_sentiment_analyses[i]
        
        polarity, subjectivity = float(textblob_analysis[0]), float(textblob_analysis[1]) 
        
        # more subjectivity -> make FinBERT's results more evenly spread across 3 categories
        # more objectivity -> make FinBERT's results more lopsided across 3 categories
        min_sentiment, max_sentiment = min(finbert_analysis), max(finbert_analysis)
        for i in range(len(finbert_analysis)):
            if finbert_analysis[i] == min_sentiment:
                finbert_analysis[i] -= 0.05 * (0.5 - subjectivity)
            elif finbert_analysis[i] == max_sentiment:
                finbert_analysis[i] += 0.05 * (0.5 - subjectivity)
        
        new_pos, new_neg, new_neu = finbert_analysis
        new_sentiments = {
            'positive': new_pos,
            'negative': new_neg,
            'neutral': new_neu,
        }
        sentiment_scores.append(new_sentiments)
    
    sentiment_scores_array = np.array([list(score.values()) for score in sentiment_scores])
    return sentiment_scores_array

def get_sentence_sentiment_ratios(
    sentence_list: List[str]
) -> Dict[str, int]:
    """
    Purpose: Calculate sentiment percentages (positive, negative, and neutral) 
        for a list of text sentences.
    :param sentence_list: List[str] representing a collection of text sentences.
    :return Dict[str, int]: A dictionary representing the sentiment percentages 
        and the number of sentences analyzed.
    """
    # compute average sentiment scores across all sentences
    sentiment_arr = _get_sentiment_analysis(
        doc = sentence_list
    )
    sentiment_arr = np.mean(sentiment_arr, axis = 0)
    pos_avg, neg_avg, neutral_avg = sentiment_arr[0], sentiment_arr[1], sentiment_arr[2]
    sentiment_summary_dict = {
        'num_articles': len(sentence_list),  # total sentences analyzed
        'positive': pos_avg,     # percentage of positive sentiment
        'negative': neg_avg,     # percentage of negative sentiment
        'neutral': neutral_avg,  # percentage of neutral sentiment
    }
    return sentiment_summary_dict

def build_compile_LSTM_RNN(
    window_size: int, 
    sentiment_cols: List[str]
) -> Sequential:
    """
    Purpose: Build and compile an LSTM (Long Short-Term Memory) model for 
        time series prediction.
    :param window_size: int representing the number of previous time steps 
        to consider in the model.
    :param sentiment_cols: List[str] representing sentiment columns for 
        input data.
    :return: An LSTM model for time series prediction.
    """
    # initialize a sequential model
    lstm_model = Sequential() 
    # add LSTM layer with 50 units, and return sequences for input data
    lstm_model.add(
        LSTM(
            50, 
            return_sequences = True, 
            input_shape = (window_size, 1 + len(sentiment_cols))
        )
    ) 
    # add another LSTM layer with 50 units and return sequences
    lstm_model.add(
        LSTM(50, return_sequences = True)
    ) 
    # add a final LSTM layer with 50 units
    lstm_model.add(LSTM(50))
    # add a dense layer with one unit for prediction
    lstm_model.add(Dense(1))
    # compile the model using mean squared error loss and the Adam optimizer
    lstm_model.compile(
        loss = 'mean_squared_error', 
        optimizer = Adam()
    )
    return lstm_model
