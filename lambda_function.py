import os
import traceback
import uuid

import boto3
from fastapi import FastAPI, HTTPException, Request
from mangum import Mangum
from pydantic import BaseModel

from api.config_parser import ConfigParser
from api.create_client import ClientFactory
from api.service.chat_service import ChatService
from api.service.doc_qa_service import DocQaProcessor

SESSION = boto3.Session(region_name="us-east-1")  # lambda doesnt recognize profiles


KEY = ConfigParser(session=SESSION).load_config(parameter_name="/openai/apikey")

open_ai_client, lang_open_ai_client, lang_open_ai_embeddings = ClientFactory.get_client(
    client="openai", api_key=KEY
)


app = FastAPI()
lambda_handler = Mangum(app)  # doesnot work in local machine , use app instead


@app.get("/random_id_generator")
async def root():
    return {"id": uuid.uuid4()}


@app.post("/chat")
async def chat(request: Request):

    body = await request.json()
    user_message = body.get("message")
    if not user_message:
        raise HTTPException(
            status_code=400,
            detail='No message provided, please provide in this format {"message":"HI, how  are you?"}',
        )
    try:
        response = ChatService.get_openai_completion(
            client=open_ai_client, prompt=user_message
        )

    except:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")
    return {"response": response, "status_code": 200}


@app.post("/pdf_qa")
async def chat(request: Request):
    body = await request.json()
    docname = body.get("docname")
    query = body.get("query")

    if not docname:
        raise HTTPException(status_code=400, detail="No docname provided")

    if not query:
        raise HTTPException(status_code=400, detail="No query provided")

    response = DocQaProcessor(
        boto_session=SESSION,
        openai_client=lang_open_ai_client,
        openai_embeddings=lang_open_ai_embeddings,
    ).process_doc_query(docname=docname.rstrip(".pdf"), query=query)

    return {"response": response, "status_code": 200}
