import time

# -----------------------------------------
# CALCULATE RESPONSE TIME
# -----------------------------------------

def calculate_latency(

    start_time

):

    return round(

        (time.time() - start_time) * 1000,

        2

    )

# -----------------------------------------
# CLEAN TEXT
# -----------------------------------------

def clean_text(text):

    return text.strip().replace("\n", " ")

# -----------------------------------------
# CONFIDENCE CHECK
# -----------------------------------------

def is_low_confidence(

    confidence,
    threshold=0.60

):

    return confidence < threshold

# -----------------------------------------
# FORMAT RESPONSE
# -----------------------------------------

def format_response(

    answer,
    intent,
    emotion

):

    return {

        "intent": intent,

        "emotion": emotion,

        "answer": answer

    }