import traceback
import uuid

from fastapi import FastAPI, HTTPException, Request

from api.chat_service import ChatService
from api.create_client import get_grok_client

# groq_client
CLIENT = get_grok_client(api_key="")


app = FastAPI()  # doesnot work in local machine , use app instead


@app.get("/random_id_generator")
async def root():
    return {"id": uuid.uuid4()}


@app.post("/chat")
async def chat(request: Request):

    # request ->>{"headers", "body":""}
    # {"message":"Hi, How are you?"} - sample body
    body = await request.json()
    user_message = body.get("message")
    if not user_message:
        raise HTTPException(
            status_code=400,
            detail='No message provided, please provide in this format {"message":"HI, how  are you?"}',
        )
    try:
        response = ChatService.get_groq_completion(client=CLIENT, prompt=user_message)

    except:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")
    return {"response": response, "status_code": 200}
