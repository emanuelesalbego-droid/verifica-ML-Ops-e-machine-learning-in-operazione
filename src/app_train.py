import torch
from transformers import pipeline
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report

def load_sentiment_model():
    """
    Carica il modello pre-addestrato per analisi del sentiment.
    """
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
    return sentiment_pipeline

def load_dataset_example():
    """
    Carica un dataset di esempio per sentiment analysis.
    Utilizziamo un dataset pubblico da Hugging Face.
    """
    # Esempio: dataset di sentiment su Twitter da tweet_eval
    dataset = load_dataset("tweet_eval", "sentiment", split="test[:100]")  # Primi 100 esempi per test
    return dataset

def preprocess_data(dataset):
    """
    Preprocessa i dati se necessario.
    Per RoBERTa, il preprocessing è gestito dalla pipeline.
    """
    texts = dataset["text"]
    labels = dataset["label"]  # Assumendo che le etichette siano 0,1,2 per negativo, neutro, positivo
    return texts, labels

def evaluate_model(pipeline, texts, labels):
    """
    Valuta il modello su testi e etichette.
    """
    predictions = []
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    for text in texts:
        result = pipeline(text)[0]
        pred_label = label_map.get(result['label'], -1)  # Mappa a numero
        predictions.append(pred_label)

    accuracy = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions, target_names=['negative', 'neutral', 'positive'])
    return accuracy, report

if __name__ == "__main__":
    # Carica modello
    sentiment_pipeline = load_sentiment_model()

    # Carica dataset
    dataset = load_dataset_example()
    texts, labels = preprocess_data(dataset)

    # Valuta modello
    accuracy, report = evaluate_model(sentiment_pipeline, texts, labels)

    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    # Esempio di inference su testo personalizzato
    test_text = "I love this new feature!"
    result = sentiment_pipeline(test_text)
    print(f"Test text: {test_text}")
    print(f"Sentiment: {result}")