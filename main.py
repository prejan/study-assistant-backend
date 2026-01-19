from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from typing import Literal
from dotenv import load_dotenv

# Load environment variables from .env (local only)
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Study Assistant API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenRouter model (OpenAI-compatible)
# IMPORTANT: API key & base URL are read from environment variables
model = OpenAIModel(
    "mistralai/mistral-7b-instruct"
)

# Agents
explain_agent = Agent(
    model,
    system_prompt="""
    You are an expert tutor who explains complex concepts in simple, clear language.
    Use analogies, examples, and break down topics into digestible parts.
    Format explanations with headings and bullet points.
    """
)

quiz_agent = Agent(
    model,
    system_prompt="""
    You are a quiz generator.
    Create 3–5 multiple choice questions with 4 options (A, B, C, D).
    Provide correct answers at the end.
    """
)

notes_agent = Agent(
    model,
    system_prompt="""
    You are a study notes generator.
    Create well-structured notes with:
    - Overview
    - Key Concepts
    - Important Points
    - Study Tips
    Use emojis and clean formatting.
    """
)

# Request & Response models
class TaskRequest(BaseModel):
    topic: str
    task_type: Literal["explain", "quiz", "notes"]

class TaskResponse(BaseModel):
    result: str
    success: bool

# Routes
@app.get("/")
async def root():
    return {
        "message": "Study Assistant API is running",
        "status": "healthy"
    }

@app.post("/generate", response_model=TaskResponse)
async def generate_content(request: TaskRequest):
    try:
        if request.task_type == "explain":
            agent = explain_agent
            prompt = f"Explain the concept of '{request.topic}' in simple terms."
        elif request.task_type == "quiz":
            agent = quiz_agent
            prompt = f"Generate quiz questions about '{request.topic}'."
        else:
            agent = notes_agent
            prompt = f"Create study notes for '{request.topic}'."

        # Run the agent
        result = await agent.run(prompt)

        # IMPORTANT FIX: use result.output (NOT result.data)
        return TaskResponse(
            result=result.output,
            success=True
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating content: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "study-assistant"
    }

# Local run (Render ignores this block)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
