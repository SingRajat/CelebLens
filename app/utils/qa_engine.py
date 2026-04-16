import os
import requests

class QAEngine:
    def __init__(self):
        self.api_key=os.getenv("GROQ_API_KEY")
        self.api_url="https://api.groq.com/openai/v1/chat/completions"
        self.model="llama-3.3-70b-versatile"

    def ask_about_celebrity(self,name,question):
        if not self.api_key:
            return "Error: GROQ_API_KEY not found. Please set it in your .env file."
        
        headers={
            "Authorization":f"Bearer {self.api_key}",
            "Content-Type":"application/json",
        }

        prompt=f"""You are an AI Assistant that knows a lot about celebrities. Answer questions about {name} concisely and accurately.

Question: {question}

Provide a detailed answer based on your knowledge about {name}."""

        payload={
            "model":self.model,
            "messages":[{"role":"user","content":prompt}],
            "temperature":0.3,
            "max_tokens":512
        }
        try:
            response=requests.post(self.api_url,headers=headers,json=payload)

            if response.status_code==200:
                return response.json()['choices'][0]['message']['content']
            else:
                error_detail = response.text
                print(f"QA API Error: {response.status_code} - {error_detail}")
                return f"Error: API returned status {response.status_code}. Details: {error_detail[:200]}"
        except Exception as e:
            print(f"Error in QA engine: {str(e)}")
            return f"Error: {str(e)}"