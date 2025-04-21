from groq import Groq
import os

# Set up the Groq client with the API key
# You can also set this as an environment variable: GROQ_API_KEY
os.environ["GROQ_API_KEY"] = "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"

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Create a completion using Meta's Llama model via Groq
completion = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/f/f2/LPU-v1-die.jpg"
                    }
                }
            ]
        }
    ],
    temperature=1,
    max_completion_tokens=1024,
    top_p=1,
    stream=False,
    stop=None,
)

# Print the response
print(completion.choices[0].message) 