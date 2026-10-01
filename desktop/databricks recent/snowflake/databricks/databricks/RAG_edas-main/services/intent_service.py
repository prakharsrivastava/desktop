from transformers import pipeline

classifier = pipeline(

    "zero-shot-classification",

    model="facebook/bart-large-mnli"

)

labels = [

    "Question",

    "Complaint",

    "Request",

    "Feedback"

]

def detect_intent(query):

    result = classifier(

        query,

        labels

    )

    return {

        "intent": result["labels"][0],

        "confidence": result["scores"][0]

    }