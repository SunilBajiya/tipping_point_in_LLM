from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai

from config import GEMINI_API_KEY, TIPPING_POINT, RECENT_MESSAGES

# ------------------------
# Gemini configuration
# ------------------------
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI()

# ------------------------
# In-memory conversation
# ------------------------
conversation = []
summary = ""


class ChatRequest(BaseModel):
    message: str


# ------------------------
# Simple summarizer
# ------------------------
def summarize_old_messages(messages):
    text = " | ".join(messages)
    return f"User discussed: {text[:300]}"


# ------------------------
# Build controlled context
# ------------------------
def build_context(new_message):
    global summary

    conversation.append(new_message)

    # 🔥 TIPPING POINT CHECK
    if len(conversation) > TIPPING_POINT:
        old_msgs = conversation[:-RECENT_MESSAGES]
        recent_msgs = conversation[-RECENT_MESSAGES:]

        summary = summarize_old_messages(old_msgs)

        context = f"""
SYSTEM:
You are a focused AI assistant.
Ignore irrelevant old context.

SUMMARY OF PREVIOUS CHAT:
{summary}

RECENT MESSAGES:
{' '.join(recent_msgs)}

INSTRUCTION:
Answer ONLY the latest user query clearly.
"""
    else:
        context = " ".join(conversation)

    return context


# ------------------------
# Chat API
# ------------------------
@app.post("/chat")
def chat(req: ChatRequest):
    context = build_context(req.message)

    response = model.generate_content(context)

    return {
        "reply": response.text,
        "total_messages": len(conversation),
        "context_sent_to_gemini": context
    }