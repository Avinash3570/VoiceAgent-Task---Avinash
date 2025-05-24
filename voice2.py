import whisper
import sounddevice as sd
import soundfile as sf
from llama_cpp import Llama
import edge_tts
import asyncio
import tempfile
import os
from playsound import playsound
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time

MODEL_PATH = "models/DeepSeek-R1-Distill-Llama-8B-Q4_0.gguf"
AUDIO_FILE = "input.wav"
PDF_PATH = "D:/Computer_Vision/Voice_AI/docs/ML_resume.pdf"

whisper_model = whisper.load_model("medium")

llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4, n_gpu_layers=0)

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
chunks = splitter.split_documents(documents)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding_model)
retriever = vectorstore.as_retriever()

def record_audio(filename, duration=15, samplerate=44100):
    print("🎙️ Listening... Please speak now (say 'exit' or 'quit' to stop).")
    time.sleep(1.5)  
    audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    sf.write(filename, audio, samplerate)
    print("✅ Audio recorded")

def transcribe_audio(filename):
    print("📝 Transcribing...")
    result = whisper_model.transcribe(filename)
    text = result["text"].strip().lower()
    return text if len(text) > 1 else None

def generate_response(user_input, retriever):
    print("🧠 Thinking...")
    docs = retriever.get_relevant_documents(user_input)[:3]
    context = "\n\n".join([doc.page_content for doc in docs])


    full_prompt = f"""
You are Nira, the AI assistant for Analytas, an AI advisory firm. Use the context below from relevant documents to answer the user's question clearly, helpfully, and conversationally. Do not mention your identity or role. Focus only on answering the question based on the context. If the answer is not in the context, gently suggest scheduling a discovery call.

Context:
{context}

User: {user_input}
Nira:"""

    output = llm(full_prompt, max_tokens=1024, stop=["User:", "Nira:"])
    return output["choices"][0]["text"].strip()

async def speak(text):
    communicate = edge_tts.Communicate(text=text, voice="en-US-AriaNeural")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        temp_path = tmp_file.name
    await communicate.save(temp_path)
    playsound(temp_path)
    os.remove(temp_path)

async def conversation_loop():
    await speak("Hello, I am Nira, your AI assistant. How can I help you today?")
    while True:
        record_audio(AUDIO_FILE)
        user_text = transcribe_audio(AUDIO_FILE)
        if not user_text:
            print("⚠️ No valid speech detected.")
            await speak("Sorry, I couldn't hear you. Could you repeat that?")
            continue

        print(f"🗣️ You said: {user_text}")

        if user_text.strip() in ["exit", "quit"]:
            print("👋 Exiting conversation.")
            await speak("Goodbye! Have a great day.")
            break

        response = generate_response(user_text, retriever)
        print(f"🤖 Assistant: {response}")
        await speak(response)

if __name__ == "__main__":
    asyncio.run(conversation_loop())
