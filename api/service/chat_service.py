# Project 1.
# We are building a web appliation that will answer all questions about their HOA using Gen AI tools like ChatGPT.
# I uploaded the HOA pdf doc to S3 bucket in Suprha account under hoabot,
# Your first task is to create a AWS API Gateway that integrates with lambda written in python, The lambda should take a parameter that is sent as a POST request and then ask ChatGPT using its API and provide the answer back to the API as response.
# The question is a string example "Do I need permission to install a flowerbed on my property from HOA?"
# The Python code will use the ChatGPT API Key that is in the systems manager parameter store at /hoabot/chatgpt/apikey

import openai


class ChatService:

    @staticmethod
    def get_openai_completion(client, prompt, model="gpt-3.5-turbo"):
        messages = [{"role": "user", "content": prompt}]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )

        return response.choices[0].message["content"]

    @staticmethod
    def get_groq_completion(client, prompt, model="llama3-8b-8192"):
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
        )

        return chat_completion.choices[0].message.content
