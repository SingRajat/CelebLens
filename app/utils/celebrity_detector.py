import os
import base64
import requests
import google.generativeai as genai

class CelebrityDetector:
    def __init__(self):
        # Load all three keys
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.hf_token = os.getenv("HF_TOKEN")
        
        self.system_prompt = """You are an elite celebrity detection AI. First, carefully analyze the facial bone structure, eye shape, nose width, lips, jawline, and hair in the image. Compare these features step-by-step to known public figures. Finally, identify the celebrity. Respond strictly in this exact format:

- **Analysis**: [your step-by-step facial feature analysis]
- **Full Name**: [celebrity name]
- **Profession**: [their profession]
- **Nationality**: [their nationality]
- **Famous For**: [what they're famous for]
- **Top achievements**: [list achievements]

If no celebrity is found, return "No celebrity found"."""

    def identify(self, image_bytes, engine="gemini"):
        try:
            if engine == "gemini":
                return self._identify_with_gemini(image_bytes)
            else:
                return self._identify_with_openai_format(image_bytes, engine)
        except Exception as e:
            print(f"Error in detection: {str(e)}")
            return f"Error: {str(e)}", ""

    def _identify_with_gemini(self, image_bytes):
        if not self.gemini_key:
            return "Error: GEMINI_API_KEY not found in .env.", ""
        
        genai.configure(api_key=self.gemini_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        image_parts = [{"mime_type": "image/jpeg", "data": image_bytes}]
        
        response = model.generate_content([self.system_prompt, image_parts[0]])
        return response.text, self.extract_name(response.text)

    def _identify_with_openai_format(self, image_bytes, engine):
        if engine == "huggingface":
            if not self.hf_token: return "Error: HF_TOKEN not found.", ""
            api_url = "https://huggingface.co/api/inference-proxy/together/v1/chat/completions" # Note: standard API router is 'https://api-inference.huggingface.co/v1/chat/completions'
            api_url = "https://api-inference.huggingface.co/v1/chat/completions"
            token = self.hf_token
            model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        else:
            if not self.groq_key: return "Error: GROQ_API_KEY not found.", ""
            api_url = "https://api.groq.com/openai/v1/chat/completions"
            token = self.groq_key
            model_name = "meta-llama/llama-4-scout-17b-16e-instruct"

        headers = {"Authorization": f"Bearer {token}"}
        encoded_image = base64.b64encode(image_bytes).decode()
        
        payload = {
            "model": model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": self.system_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]
            }],
            "temperature": 0.3, 
            "max_tokens": 1024
        }
        
        response = requests.post(api_url, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()['choices'][0]['message']['content']
            return result, self.extract_name(result)
        else:
            return f"Error: API Request Failed: {response.text[:200]}", ""

    def extract_name(self, content):
        for line in content.splitlines():
            if line.lower().startswith("- **full name**:"):
                return line.split(":")[1].strip()
        return "unknown"
