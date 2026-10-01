from fastapi import FastAPI

from routes.upload_api import router as upload_router

from routes.query_api import router as query_router

# ---------------------------------------------------
# CREATE FASTAPI APP
# ---------------------------------------------------

app = FastAPI(

    title="DocAssist+",

    version="1.0"

)

# ---------------------------------------------------
# REGISTER ROUTES
# ---------------------------------------------------

app.include_router(upload_router)

app.include_router(query_router)

# ---------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------

@app.get("/")

def health_check():

    return {

        "status": "running",

        "service": "DocAssist+"

    }