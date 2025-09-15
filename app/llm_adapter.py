import os, asyncio
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

GEMINI_TEXT_MODEL = os.getenv('GEMINI_TEXT_MODEL','gemini-2.5-flash')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not GEMINI_API_KEY:
    raise RuntimeError('GEMINI_API_KEY required')

# Create a client instance
client = genai.Client(api_key=GEMINI_API_KEY)

async def generate_answer_gemini_stream(question: str, retrieved: list):
    prompt = "You are a helpful assistant. Use the context to answer the question.\n\nContext:\n"
    for r in retrieved:
        prompt += r.get('text','')[:1000] + "\n---\n"
    prompt += "\nQuestion: " + question + "\nAnswer:" 
    
    try:
        # Use the new Google GenAI SDK for streaming
        response = client.models.generate_content_stream(
            model=GEMINI_TEXT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=512,
                temperature=0.7,
            )
        )
        
        # Stream the response chunks
        for chunk in response:
            if chunk.text:
                for ch in chunk.text:
                    await asyncio.sleep(0.002)
                    yield ch
                    
    except Exception as e:
        # Fallback to non-streaming if streaming fails
        try:
            response = client.models.generate_content(
                model=GEMINI_TEXT_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=512,
                    temperature=0.7,
                )
            )
            text = response.text if response.text else "Sorry, I couldn't generate a response."
            
            for ch in text:
                await asyncio.sleep(0.002)
                yield ch
        except Exception as fallback_e:
            error_text = f"Error generating response: {str(e)}"
            for ch in error_text:
                await asyncio.sleep(0.002)
                yield ch
