import requests

def talk_to_brain(prompt):
    # LM Studio uses the /v1/chat/completions endpoint
    url = "http://localhost:1234/v1/chat/completions"
    
    data = {
        "model": "google/gemma-4-e4b", # This name just needs to be a string for LM Studio
        "messages": [
            {"role": "system", "content": "You are a professional business assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    
    response = requests.post(url, json=data)
    
    # LM Studio/OpenAI format uses ['choices'][0]['message']['content']
    result = response.json()
    return result['choices'][0]['message']['content']

# Example Business Task
try:
    print(talk_to_brain("Summarize this legal memo: The company is planning for a merger."))
except Exception as e:
    print(f"Error: Make sure LM Studio 'Local Server' is STARTED. Details: {e}")
