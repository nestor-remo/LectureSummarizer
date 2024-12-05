from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment
from pydub.utils import make_chunks
import os

load_dotenv()

client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)

def transcribe_audio(audio_file):
    try:
        audio_file = open(audio_file, "rb")
        transcription = client.audio.transcriptions.create(
            model = "whisper-1",
            file = audio_file,
            response_format = "text"
        )
        return transcription
    except Exception as e:
        print(f"Error: {e}")
        return None
    
def transcribe_audio_chunk(chunk, index):
    chunk_file = f"chunk_{index}.mp4"
    chunk.export(chunk_file, format="mp4")

    transcription = transcribe_audio(chunk_file)
    if transcription:
        return transcription
    return None
    
def transcribe_large_file(audio_file, chunk_size= 10 * 60 * 1000):
    audio = AudioSegment.from_file(audio_file, format="mp4")
    chunks = make_chunks(audio, chunk_size)
    transcriptions = []

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)}...")
        transcription = transcribe_audio_chunk(chunk, i)
        transcriptions.append(transcription)

    full_transcription = " ".join(transcriptions)
    return full_transcription

def summarize(text):
    try:
        response = client.chat.completions.create(
            messages = [
                { "role": "system", "content": "You are a helpful assistant." },
                {
                    "role": "user",
                    "content": f"Summarize the following text: {text}"
                }
            ],
            model = "gpt-4o-mini",
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in summarize: {e}")
        return None

if __name__ == "__main__":
    audio_file = "lecture.mp4"

    transcription = transcribe_large_file(audio_file)
    if transcription:
        print(f"Transcription: {transcription}")

        with open("transcription.txt", "w") as file:
            file.write(transcription)
            print("Transcription saved to transcription.txt")

        summary = summarize(transcription)
        if summary:
            print(f"Summary: {summary}")

        with open("summary.txt", "w") as file:
            file.write(summary)
            print("Summary saved to summary.txt")
        