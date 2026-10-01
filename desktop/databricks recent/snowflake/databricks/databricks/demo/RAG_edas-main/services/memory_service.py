import redis
import json

# ---------------------------------------------------
# REDIS CONNECTION
# ---------------------------------------------------

redis_client = redis.Redis(

    host="localhost",
    port=6379,
    decode_responses=True

)

# ---------------------------------------------------
# SAVE SESSION
# ---------------------------------------------------

def save_session(

    session_id,
    data

):

    redis_client.set(

        session_id,

        json.dumps(data)

    )

# ---------------------------------------------------
# GET SESSION
# ---------------------------------------------------

def get_session(session_id):

    data = redis_client.get(session_id)

    if data:

        return json.loads(data)

    return {}

# ---------------------------------------------------
# SLOT FILLING
# ---------------------------------------------------

def handle_slot_filling(

    session,
    user_query

):

    # -----------------------------------------------
    # CHECK LOAN STATUS FLOW
    # -----------------------------------------------

    if session.get("intent") == "loan_status":

        # If loan_id missing
        if not session.get("loan_id"):

            session["loan_id"] = user_query

            save_session(

                session["session_id"],

                session

            )

            return "Your loan is approved"

    return None