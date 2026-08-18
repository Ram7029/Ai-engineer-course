import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key kaha hai bhai")

client=Groq(api_key=my_api_key)

model="llama-3.3-70b-versatile"


#step1->Knowledgw base

knowledge_base={
    "age" : "The age of Aritesh is 20 years.",
    "hobby" : "Aritesh loves to play cricket.",
    "networth" : "Aritesh has a networth of 1 million dollars."
}

#step2 -> retrival

def retrieve_info(question):
    if "age" in question.lower():
        return knowledge_base["age"]
    elif "hobby" in question.lower():
        return knowledge_base["hobby"]
    elif "networth" in question.lower():
        return knowledge_base["networth"]
    else:
        return "Sorry, I don't have information about that."

def ask_llm(question):
    content=retrieve_info(question)
    sys_prompt=f"""Answer the questions in one line only. AND answer only based on the context provided. Do not hallucinate. If the answer is not in the context, say 'Sorry, I don't have information about that.'Context: {content}"""
    system_message={
        "role": "system",
        "content": sys_prompt
    }
    message={
        "role": "user",
        "content": question
    }
    messages=[system_message, message]
    response=client.chat.completions.create(model=model, messages=messages)
    answer=response.choices[0].message.content
    return answer

question="What is Aritesh's networth?"
print(ask_llm(question))

