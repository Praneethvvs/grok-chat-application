import os
from openai import OpenAI
from groq import Groq


def get_openai_client(api_key):
    client = OpenAI(
        # This is the default and can be omitted
        api_key=api_key,
    )
    return client


def get_grok_client(api_key):
    client = Groq(
        api_key=api_key,
    )
    return client
