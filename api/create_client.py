import os

from groq import Groq
from langchain_openai import OpenAI as LangOpenAI
from langchain_openai import OpenAIEmbeddings as LangOpenAIEmbeddings
from openai import OpenAI


class ClientFactory:

    def get_client(client, api_key):
        if client == "openai":
            return (
                OpenAI(api_key=api_key),
                LangOpenAI(api_key=api_key),
                LangOpenAIEmbeddings(api_key=api_key),
            )

        elif client == "grok":
            return Groq(api_key=api_key)
        else:
            raise ValueError("Incorrect client: Accepted values ['openai', 'grok']")
