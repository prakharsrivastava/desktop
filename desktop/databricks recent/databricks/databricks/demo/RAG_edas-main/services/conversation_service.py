def detect_context_switch(

    session,
    user_query

):

    if "complaint" in user_query.lower():

        session["intent"] = "complaint"

        session["loan_id"] = None

        return True

    return False