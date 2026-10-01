from transformers import pipeline

emotion_classifier = pipeline(

    "text-classification",

    model="j-hartmann/emotion-english-distilroberta-base"

)

def detect_emotion(query):

    result = emotion_classifier(query)

    return result[0]