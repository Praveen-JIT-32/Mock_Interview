import json
import boto3
import difflib
import uuid
import asyncio
import re
import contextlib
from datetime import datetime
from typing import Optional, List, Dict, Any
import numpy as np
import base64
import cv2
import mediapipe as mp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Form, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool
from botocore.exceptions import ClientError, BotoCoreError
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
import logging
from enum import Enum
from dataclasses import dataclass

# ================== FASTAPI APPLICATION ==================
app = FastAPI(
    title="Production Interview API",
    description="AI-powered interview system with comprehensive evaluation and face monitoring",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ================== CONFIGURATION ==================
class Config:
    # AWS Configuration
    S3_BUCKET = "mock_interview25"
    S3_FOLDER = "interviewphoto"
    AWS_REGION = "us-east-1"
    BEDROCK_MODEL_ID = "amazon.nova-pro-v1:0"
    
    # Database Configuration
    DB_HOST = "localhost"
    DB_NAME = "vijay"
    DB_USER = "postgres"
    DB_PASSWORD = "Arnold@123"
    DB_PORT = 5432
    DB_MIN_CONNECTIONS = 2
    DB_MAX_CONNECTIONS = 10
    
    # Interview Configuration
    MAX_QUESTIONS = 5
    SAMPLE_RATE = 16000
    CODING_QUESTION_NUMBERS = [4, 5]
    
    # Timing Configuration
    INITIAL_WAIT_LIMIT = 7.0
    SPEAKING_WINDOW = 180.0
    SILENCE_TIMEOUT = 5.0 
    CODING_TIME_LIMIT = 600.0
    FACE_CHECK_INTERVAL = 5.0
    
    # Scoring Configuration
    MAX_SCORE = 10.0
    POINTS_PER_QUESTION = 2.0
    RELEVANCE_WEIGHT = 0.7
    GRAMMAR_WEIGHT = 0.3
    PENALTY_PER_VIOLATION = 0.5
    MAX_PENALTY = 3.0
    
    AUDIO_CHUNK_SIZE = 3200


class InterviewMode(str, Enum):
    BASIC = "Basic"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"


class FaceViolationType(str, Enum):
    NO_FACE = "no_face"
    MULTIPLE_FACES = "multiple_faces"

# ================== LOGGING ==================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)-8s] [%(name)-25s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ================== GLOBAL CLIENTS ==================
def initialize_aws_clients():
    """Initialize AWS clients once"""
    try:
        return {
            's3': boto3.client("s3", region_name=Config.AWS_REGION),
            'textract': boto3.client("textract", region_name=Config.AWS_REGION),
            'polly': boto3.client("polly", region_name=Config.AWS_REGION),
            'comprehend': boto3.client("comprehend", region_name=Config.AWS_REGION),
            'bedrock': boto3.client("bedrock-runtime", region_name=Config.AWS_REGION)
        }
    except Exception as e:
        logger.critical(f"Failed to initialize AWS clients: {e}")
        raise

aws_clients = initialize_aws_clients()
logger.info("AWS clients initialized successfully")

# ================== DATABASE POOL ==================
def initialize_db_pool():
    """Initialize database connection pool"""
    try:
        pool = ThreadedConnectionPool(
            Config.DB_MIN_CONNECTIONS,
            Config.DB_MAX_CONNECTIONS,
            host=Config.DB_HOST,
            database=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            port=Config.DB_PORT
        )
        logger.info("Database connection pool initialized successfully")
        return pool
    except Exception as e:
        logger.critical(f"Failed to initialize database pool: {e}")
        raise

db_pool = initialize_db_pool()

# ================== DATABASE OPERATIONS ==================
async def execute_query(query: str, params: tuple = None, fetch: bool = False):
    """Execute database query"""
    conn = None
    try:
        conn = db_pool.getconn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            if fetch:
                result = cur.fetchall() if fetch == "all" else cur.fetchone()
                return result
            conn.commit()
            return True
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database query failed: {e}")
        raise
    finally:
        if conn:
            db_pool.putconn(conn)

# ================== SESSION OPERATIONS ==================
async def create_session(email: str) -> str:
    """Create a new interview session"""
    session_id = str(uuid.uuid4())
    query = """
        INSERT INTO session (id, email, created_at)
        VALUES (%s, %s, %s)
    """
    try:
        await execute_query(query, (session_id, email, datetime.now()))
        logger.info(f"Created session {session_id} for {email}")
        return session_id
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create session"
        )

async def get_session(session_id: str) -> Optional[Dict]:
    """Retrieve session by ID"""
    query = "SELECT id, email, created_at FROM session WHERE id = %s"
    try:
        row = await execute_query(query, (session_id,), fetch=True)
        return row if row else None
    except Exception as e:
        logger.error(f"Failed to get session: {e}")
        return None

async def validate_session(session_id: str, email: str) -> Dict:
    """Validate session exists and matches email"""
    session = await get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    if session['email'] != email:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email does not match session"
        )
    return session

# ================== STUDENT OPERATIONS ==================
async def get_student(email: str) -> Optional[Dict]:
    """Get student data by email"""
    query = "SELECT reg_number, email FROM student WHERE email = %s"
    try:
        row = await execute_query(query, (email,), fetch=True)
        return row if row else None
    except Exception as e:
        logger.error(f"Failed to get student: {e}")
        return None

# ================== TOKEN OPERATIONS ==================
async def deduct_token(email: str) -> int:
    """Deduct one interview token from student"""
    conn = None
    try:
        conn = db_pool.getconn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT tokens_remaining FROM student_tokens WHERE email = %s FOR UPDATE",
                (email,)
            )
            row = cur.fetchone()
            
            if not row:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Token record not found"
                )
            
            tokens = row["tokens_remaining"] or 0
            if tokens <= 0:
                raise HTTPException(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    detail="No interview tokens remaining"
                )
            
            new_balance = tokens - 1
            cur.execute(
                """
                UPDATE student_tokens
                SET tokens_remaining = %s, last_updated = NOW()
                WHERE email = %s
                """,
                (new_balance, email)
            )
            conn.commit()
            logger.info(f"Deducted token for {email}, new balance: {new_balance}")
            return new_balance
    except HTTPException:
        if conn:
            conn.rollback()
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Token deduction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token deduction failed"
        )
    finally:
        if conn:
            db_pool.putconn(conn)

# ================== INTERACTION OPERATIONS ==================
async def save_interaction(
    candidate_email: str,
    agent_text: str,
    candidate_text: str = "",
    question_number: int = 1,
    session_id: Optional[str] = None,
    skill: str = "",
    mode: str = ""
) -> str:
    """Save interview interaction"""
    item_id = str(uuid.uuid4())
    student = await get_student(candidate_email)
    reg_number = student['reg_number'] if student else None
    
    query = """
        INSERT INTO interaction_table 
        (id, session_id, candidate_email, agent_speak, candidate_speak, 
         question_number, reg_number, skill, mode)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    try:
        await execute_query(
            query,
            (item_id, session_id, candidate_email, agent_text, candidate_text,
             question_number, reg_number, skill, mode)
        )
        logger.info(f"Saved interaction {item_id} for session {session_id}")
        return item_id
    except Exception as e:
        logger.error(f"Failed to save interaction: {e}")
        raise

async def update_candidate_response(item_id: str, candidate_text: str):
    """Update candidate response in interaction"""
    query = """
        UPDATE interaction_table
        SET candidate_speak = %s
        WHERE id = %s
    """
    try:
        await execute_query(query, (candidate_text, item_id))
        logger.info(f"Updated candidate response for interaction {item_id}")
    except Exception as e:
        logger.error(f"Failed to update candidate response: {e}")
        raise

async def get_session_interactions(session_id: str, email: str) -> List[Dict]:
    """Get all interactions for a session"""
    query = """
        SELECT id, session_id, agent_speak, candidate_email, candidate_speak, 
               question_number, timestamp, skill, reg_number, mode
        FROM interaction_table
        WHERE candidate_email = %s AND session_id = %s
        ORDER BY question_number, timestamp ASC
        LIMIT %s
    """
    try:
        rows = await execute_query(
            query,
            (email, session_id, Config.MAX_QUESTIONS),
            fetch="all"
        )
        return rows or []
    except Exception as e:
        logger.error(f"Failed to get interactions: {e}")
        raise

# ================== VIOLATION TRACKING (IN-MEMORY) ==================
# Store violations in memory during interview
violation_trackers = {}

def get_violation_tracker(session_id: str) -> dict:
    """Get or create violation tracker for session"""
    if session_id not in violation_trackers:
        violation_trackers[session_id] = {
            'count': 0,
            'last_violation_time': 0,
            'cooldown': 3.0  # 3 seconds cooldown between violations
        }
    return violation_trackers[session_id]

def increment_violation(session_id: str) -> int:
    """Increment violation count for session"""
    tracker = get_violation_tracker(session_id)
    tracker['count'] += 1
    tracker['last_violation_time'] = asyncio.get_event_loop().time()
    logger.info(f"Violation #{tracker['count']} recorded for session {session_id}")
    return tracker['count']

def get_violation_count_memory(session_id: str) -> int:
    """Get violation count from memory"""
    tracker = violation_trackers.get(session_id, {})
    return tracker.get('count', 0)

def clear_violation_tracker(session_id: str):
    """Clear violation tracker after interview"""
    if session_id in violation_trackers:
        del violation_trackers[session_id]

# ================== SCORE OPERATIONS ==================
async def save_interview_score(
    session_id: str,
    reg_number: Optional[str],
    email: str,
    score: float,
    max_score: float,
    questions_asked: int,
    answered: int,
    relevance_feedback: str,
    focus_feedback: str,
    overall_summary: str,
    skill: str = "",
    communication_strength: str = "",
    technical_strength: str = "",
    mode: str = "",
    violation_count: int = 0
) -> bool:
    """Save interview score with violation count"""
    conn = None
    try:
        conn = db_pool.getconn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM interviewscore WHERE session_id = %s",
                (session_id,)
            )
            existing = cur.fetchone()
            
            if existing:
                cur.execute("""
                    UPDATE interviewscore SET
                        reg_number = %s, email = %s, score = %s, max_score = %s,
                        questions_asked = %s, answered = %s, relevance_feedback = %s,
                        focus_feedback = %s, overall_summary = %s, skill = %s,
                        communication_strength = %s, technical_strength = %s,
                        mode = %s, violation_count = %s,
                        evaluated_at = CURRENT_TIMESTAMP
                    WHERE session_id = %s
                """, (
                    reg_number, email, round(score, 2), max_score, questions_asked,
                    answered, relevance_feedback, focus_feedback, overall_summary,
                    skill, communication_strength, technical_strength, mode,
                    violation_count, session_id
                ))
            else:
                cur.execute("""
                    INSERT INTO interviewscore (
                        session_id, reg_number, email, score, max_score,
                        questions_asked, answered, relevance_feedback,
                        focus_feedback, overall_summary, skill,
                        communication_strength, technical_strength, mode,
                        violation_count
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    session_id, reg_number, email, round(score, 2), max_score,
                    questions_asked, answered, relevance_feedback, focus_feedback,
                    overall_summary, skill, communication_strength,
                    technical_strength, mode, violation_count
                ))
            
            conn.commit()
            logger.info(f"Saved interview score for session {session_id} with {violation_count} violations")
            return True
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Failed to save interview score: {e}")
        return False
    finally:
        if conn:
            db_pool.putconn(conn)

# ================== AI SERVICES ==================
async def invoke_bedrock(prompt: str, max_tokens: int = 1000, temperature: float = 0.8) -> str:
    """Invoke Bedrock model"""
    try:
        response = aws_clients['bedrock'].invoke_model(
            modelId=Config.BEDROCK_MODEL_ID,
            contentType='application/json',
            accept='application/json',
            body=json.dumps({
                "messages": [{
                    "role": "user",
                    "content": [{"text": prompt}]
                }],
                "inferenceConfig": {
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                    "topP": 0.9
                }
            })
        )
        
        response_body = json.loads(response['body'].read())
        text = (
            response_body
            .get("output", {})
            .get("message", {})
            .get("content", [{}])[0]
            .get("text", "")
            .strip()
        )
        
        if not text:
            raise ValueError("Empty response from Bedrock")
        
        return text
    except (ClientError, BotoCoreError) as e:
        logger.error(f"Bedrock API error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service temporarily unavailable"
        )
    except Exception as e:
        logger.error(f"Unexpected error in Bedrock invocation: {e}")
        raise

# ================== QUESTION GENERATION ==================
def detect_coding_language(skills: str) -> str:
    """Detect primary coding language from skills"""
    if not skills:
        return "Python"
    
    skill_lower = skills.lower()
    languages = [
        "python", "javascript", "java", "c++", "c#", "typescript",
        "c", "go", "rust", "php", "swift", "kotlin","node.js","dotnet",
        "typescript", "sql", "html", "css" ,"mongodb","react.js","angular" 
    ]
    
    for lang in languages:
        if lang in skill_lower:
            return lang.capitalize()
    
    return "Python"

# ================== QUESTION GENERATION (DYNAMIC AI-POWERED) ==================

async def generate_non_coding_question(
    conversation: List[Dict],
    skills: str,
    question_number: int,
    mode: str
) -> str:
    """Generate non-coding interview question dynamically based on mode"""
    skills_text = skills or "technical skills"
    
    # Build conversation history context
    previous_questions = []
    if conversation:
        previous_questions = [
            msg["content"] for msg in conversation 
            if msg.get("role") == "assistant"
        ]
    
    history_context = ""
    if previous_questions:
        history_context = f"\nPreviously asked questions:\n" + "\n".join(f"- {q}" for q in previous_questions[-2:])
    
    # Question-specific prompts with dynamic mode interpretation
    prompts = {
        1: f"""You are an expert technical interviewer. Generate a professional opening question that:
- Welcomes the candidate warmly
- Asks about their experience with {skills_text}
- Is appropriate for any difficulty level
- Keep it to 2-3 lines
- Start the interview with " Welcome to the interview!"

Generate the question now:""",
              
        2: f"""You are an expert technical interviewer conducting a {mode}-level interview.

SKILL AREA: {skills_text}
DIFFICULTY: {mode}
QUESTION TYPE: Technical question requiring real-world experience explanation

Instructions:
- For {mode} level, adjust the depth and complexity accordingly
- If {mode} is "Basic", ask about fundamental concepts and simple usage
- If {mode} is "Intermediate", ask about practical applications and common scenarios
- If {mode} is "Advanced", ask about complex implementations, optimizations, or architectural decisions
- Keep it to 2-3 lines maximum
- Make it different from previous questions{history_context}

Generate a {mode}-level technical question now:""",

        3: f"""You are an expert technical interviewer conducting a {mode}-level interview.

SKILL AREA: {skills_text}
DIFFICULTY: {mode}
QUESTION TYPE: Scenario-based problem-solving question

Instructions:
- For {mode} level, adjust scenario complexity accordingly
- If {mode} is "Basic", present a simple, straightforward scenario
- If {mode} is "Intermediate", present a realistic production scenario with moderate complexity
- If {mode} is "Advanced", present a complex scenario requiring architectural thinking and trade-off analysis
- Keep it to 2-3 lines maximum
- Make it different from previous questions{history_context}

Generate a {mode}-level scenario question now:"""
    }
    
    prompt = prompts.get(question_number, prompts[3])
    
    try:
        question = await invoke_bedrock(prompt, max_tokens=250, temperature=0.7)
        
        # Check similarity with last question
        if conversation:
            last_question = next(
                (m["content"] for m in reversed(conversation) if m.get("role") == "assistant"),
                ""
            )
            if last_question:
                similarity = difflib.SequenceMatcher(None, question, last_question).ratio()
                if similarity >= 0.7:
                    # Regenerate with explicit instruction to be different
                    retry_prompt = prompt + f"\n\nIMPORTANT: Make this VERY different from: '{last_question}'"
                    question = await invoke_bedrock(retry_prompt, max_tokens=250, temperature=0.8)
        
        return question.strip()
    except Exception as e:
        logger.error(f"Failed to generate question: {e}")
        raise



# async def generate_coding_question(
#     skills: str, 
#     mode: str,
#     question_number: int
# ) -> str:
#     """Generate coding challenge with different focus based on question number"""
#     coding_language = detect_coding_language(skills)
    
  
#     # ✅ Add uniqueness seed
#     import random
#     seed = random.randint(1000, 9999)
    
#     prompt = f"""You are an expert technical interviewer creating a {mode}-level coding challenge.

# PROGRAMMING LANGUAGE: {skills}
# DIFFICULTY LEVEL: {mode}
# SKILL AREA: {skills}
# UNIQUENESS SEED: {seed}

# Instructions:
# - Create a {mode}-level problem that tests
# - For "Basic": Simple problem with clear logic, no complex algorithms needed
# - For "Intermediate": Moderate problem requiring standard
# - For "Advanced": Complex problem requiring advanced techniques
# - Generate the coding question for to run in sandbox IDE

# Problem Requirements:
# 1. Clear, concise problem statement (3-5 lines)
# 2. Input format specification
# 3. Output format specification
# 4. 1 sample test cases with expected output
# 5. Should be solvable in 10-15 minutes for a {mode}-level candidate

# 6. DO NOT include solution, hints, or implementation details
# 7. Make it creative and interesting, not a textbook problem

# Generate the {mode}-level {skills} coding problem:"""
    
#     try:
#         question = await invoke_bedrock(prompt, max_tokens=400, temperature=0.9)
#         return f"This is a {mode}-level coding question. You have 10 minutes to write your code.\n\n{question.strip()}"
#     except Exception as e:
#         logger.error(f"Failed to generate coding question: {e}")
#         raise

async def generate_coding_question(
    skills: str, 
    mode: str,
    question_number: int,
    conversation: list = None
) -> str:
    """Generate coding challenge with uniqueness and duplicate prevention."""
    
    coding_language = detect_coding_language(skills)

    import random
    seed = random.randint(1000, 9999)
    
    prompt = f"""You are an expert coding interviewer.

Create a {mode}-level coding problem that the candidate will implement in an online sandbox IDE without external configuration.

Coding Topic Area: {skills}
DIFFICULTY LEVEL: {mode}
TOPIC CONTEXT: {skills}
UNIQUENESS SEED: {seed}

Instructions:
- The question MUST be fully executable locally with no external dependencies, no cloud connections, no internet access, and no credentials.
- The code should not require AWS, external APIs, online services, or special SDK configuration.
- For Basic: simple logic, no advanced algorithms
- For Intermediate: real-world scenario with moderate complexity
- For Advanced: algorithmic thinking or data structures

Problem Requirements:
1. Clear, concise problem statement (3–5 lines)
2. Input format
3. Output format
4. EXACTLY one sample input and expected output
5. DO NOT include solution or hints

Generate the problem now:"""
    
    try:
        question = (await invoke_bedrock(prompt, max_tokens=400, temperature=0.9)).strip()

        # Duplicate prevention check
        if conversation:
            last_question = next(
                (msg["content"] for msg in reversed(conversation) if msg.get("role") == "assistant"),
                ""
            )
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, question, last_question).ratio()

            if similarity >= 0.70:
                retry_prompt = prompt + f"\n\nIMPORTANT: Make it completely different from:\n{last_question}"
                question = (await invoke_bedrock(retry_prompt, max_tokens=450, temperature=1.1)).strip()

        return f"This is a {mode}-level coding question. You have 10 minutes to write {skills} code.\n\n{question}"

    except Exception as e:
        logger.error(f"Failed to generate coding question: {e}")
        raise


# ================== EVALUATION ==================
async def score_response(question: str, answer: str) -> float:
    """Score a single response"""
    if not answer or answer in ["[no response]", ""]:
        return 0.0
    
    if len(answer.split()) <= 2:
        words = answer.split()
        valid_words = sum(1 for w in words if len(w) > 2 and w.isalpha())
        if valid_words == 0:
            return 0.1
    
    relevance_prompt = f"""
Rate the relevance of this answer to the question on a scale of 0.0 to 1.0.

Question: {question}
Answer: {answer}

Consider:
- Direct relevance (addresses the question)
- Substance (meaningful content)
- Coherence (logical flow)
- Specificity (concrete details)

Return ONLY a number between 0.0 and 1.0.
"""
    
    grammar_prompt = f"""
Rate the grammar and language quality of this text on a scale of 0.0 to 1.0.

Text: {answer}

Consider:
- Grammar accuracy
- Real words used meaningfully
- Proper sentence structure

Return ONLY a number between 0.0 and 1.0.
"""
    
    try:
        relevance_text = await invoke_bedrock(relevance_prompt, max_tokens=50)
        grammar_text = await invoke_bedrock(grammar_prompt, max_tokens=50)
        
        rel_numbers = re.findall(r"\d*\.?\d+", relevance_text)
        gram_numbers = re.findall(r"\d*\.?\d+", grammar_text)
        
        relevance = float(rel_numbers[0]) if rel_numbers else 0.0
        grammar = float(gram_numbers[0]) if gram_numbers else 0.0
        
        relevance = max(0.0, min(relevance, 1.0))
        grammar = max(0.0, min(grammar, 1.0))
        
        final_score = (
            relevance * Config.RELEVANCE_WEIGHT +
            grammar * Config.GRAMMAR_WEIGHT
        ) * Config.POINTS_PER_QUESTION
        
        return round(final_score, 2)
    except Exception as e:
        logger.error(f"Scoring failed: {e}")
        return 0.0

def parse_feedback(
    feedback_text: str,
    total_score: float,
    answered_count: int,
    violation_count: int,
    penalty: float
) -> Dict[str, str]:
    """Parse feedback into components"""
    # Score-based feedback templates
    if total_score >= 8.0:
        category = "excellent"
    elif total_score >= 6.0:
        category = "good"
    elif total_score >= 4.0:
        category = "moderate"
    else:
        category = "poor"
    
    templates = {
        "excellent": {
            "relevance": "Excellent relevance - consistently addressed questions with specific examples.",
            "focus": "Outstanding focus - maintained complete attention throughout.",
            "communication": "Exceptional communication - clear, articulate, professional delivery.",
            "technical": "Advanced technical knowledge - demonstrates deep understanding.",
            "summary": f"Top performer ({total_score:.1f}/10) - highly recommended."
        },
        "good": {
            "relevance": "Good relevance - answers generally on-topic with relevant examples.",
            "focus": "Good focus - maintained attention with minor lapses.",
            "communication": "Strong communication - clear with good technical explanation.",
            "technical": "Solid technical foundation - good understanding of core concepts.",
            "summary": f"Strong candidate ({total_score:.1f}/10) - suitable for mid-level roles."
        },
        "moderate": {
            "relevance": "Moderate relevance - some answers lacked specific connections.",
            "focus": "Inconsistent focus - showed periods of distraction.",
            "communication": "Developing communication - needs improvement in clarity.",
            "technical": "Basic technical understanding - significant gaps present.",
            "summary": f"Developing candidate ({total_score:.1f}/10) - requires improvement."
        },
        "poor": {
            "relevance": "Poor relevance - answers failed to address questions directly.",
            "focus": "Poor focus - multiple attention issues observed.",
            "communication": "Limited communication - unclear expression throughout.",
            "technical": "Minimal technical knowledge - not suitable for technical roles.",
            "summary": f"Needs significant development ({total_score:.1f}/10) - not recommended."
        }
    }
    
    feedback = templates[category]
    
    # Adjust focus based on violations
    if violation_count == 0:
        feedback["focus"] = "Excellent focus - no behavioral alerts detected."
    elif violation_count > 3:
        feedback["focus"] += f" Multiple attention issues ({violation_count} violations) detected."
    
    # Add penalty information to summary if applicable
    if penalty > 0:
        feedback["summary"] += f" Penalty of {penalty:.1f} points deducted due to {violation_count} violation(s)."
    
    return {
        "relevance_feedback": feedback["relevance"],
        "focus_feedback": feedback["focus"],
        "communication_strength": feedback["communication"],
        "technical_strength": feedback["technical"],
        "overall_summary": feedback["summary"]
    }

async def generate_feedback(
    interactions: List[Dict],
    total_score: float,
    answered_count: int,
    violation_count: int,
    penalty: float,
    skills: str
) -> Dict[str, str]:
    """Generate comprehensive interview feedback"""
    conversation_context = ""
    for inter in interactions:
        agent_text = inter.get('agent_speak', '') or ''
        candidate_text = inter.get('candidate_speak', '') or ''
        if candidate_text and candidate_text not in ["[no response]", ""]:
            conversation_context += f"Q: {agent_text}\nA: {candidate_text}\n\n"
    
    feedback_prompt = f"""
Based on this interview, provide detailed feedback:

SKILLS ASSESSED: {skills}
FINAL SCORE: {total_score}/{Config.MAX_SCORE}
QUESTIONS ANSWERED: {answered_count}/{Config.MAX_QUESTIONS}
FOCUS VIOLATIONS: {violation_count}
PENALTY DEDUCTED: {penalty} points (1 point per violation)

CONVERSATION:
{conversation_context}

Provide analysis in these areas (be honest and critical):
1. RELEVANCE FEEDBACK: (20-30 words)
2. FOCUS FEEDBACK: (20-30 words - mention violations if any)
3. COMMUNICATION STRENGTH: (20-30 words)
4. TECHNICAL STRENGTH: (20-30 words)
5. OVERALL SUMMARY: (30-40 words - mention penalty if violations occurred)
"""
    
    try:
        feedback_text = await invoke_bedrock(feedback_prompt, max_tokens=1000)
        return parse_feedback(feedback_text, total_score, answered_count, violation_count, penalty)
    except Exception as e:
        logger.error(f"Feedback generation failed: {e}")
        return generate_default_feedback(total_score, answered_count, violation_count, penalty)

def generate_default_feedback(
    total_score: float,
    answered_count: int,
    violation_count: int,
    face_penalty: float
) -> Dict[str, str]:
    """Generate default feedback when AI fails"""
    return parse_feedback("", total_score, answered_count, violation_count, face_penalty)

# ================== AUDIO PROCESSING ==================
def text_to_speech(text: str) -> bytes:
    """Convert text to speech using AWS Polly"""
    try:
        response = aws_clients['polly'].synthesize_speech(
            OutputFormat="mp3",
            VoiceId="Joanna",
            Engine="neural",
            Text=text[:2500]
        )
        return response["AudioStream"].read()
    except ClientError as e:
        logger.error(f"Polly TTS failed: {e}")
        return b""

async def stream_audio_to_transcribe(input_stream, audio_queue: asyncio.Queue):
    """Stream audio chunks to AWS Transcribe"""
    buffer = bytearray()
    chunk_size = Config.AUDIO_CHUNK_SIZE
    
    try:
        while True:
            raw_chunk = await audio_queue.get()
            if raw_chunk is None:
                if buffer:
                    try:
                        await input_stream.send_audio_event(audio_chunk=bytes(buffer))
                    except Exception:
                        pass
                    buffer.clear()
                break
            
            if isinstance(raw_chunk, (bytes, bytearray)):
                buffer.extend(raw_chunk)
            else:
                continue
            
            while len(buffer) >= chunk_size:
                chunk = bytes(buffer[:chunk_size])
                try:
                    await input_stream.send_audio_event(audio_chunk=chunk)
                except Exception:
                    pass
                del buffer[:chunk_size]
        
        try:
            await input_stream.end_stream()
        except Exception:
            pass
    except asyncio.CancelledError:
        try:
            await input_stream.end_stream()
        except Exception:
            pass

# ================== CODE EXECUTION ==================
async def run_code(code: str) -> str:
    """Execute code in sandboxed environment"""
    buffer = io.StringIO()
    try:
        wrapped_code = "def __candidate_func__():\n"
        for line in code.splitlines():
            wrapped_code += f"    {line}\n"
        wrapped_code += "\n__candidate_func__()"
        
        with contextlib.redirect_stdout(buffer):
            exec(wrapped_code, {"__builtins__": __builtins__}, {})
        
        output = buffer.getvalue().strip()
        return output or "Code executed successfully (no output)."
    except Exception as e:
        return f"Error: {e}"
    finally:
        buffer.close()

class TranscriptionHandler(TranscriptResultStreamHandler):

    def __init__(self, result_stream, ws: WebSocket):
        super().__init__(result_stream)
        self.ws = ws
        self.buffer = ""  # Stores cumulative FINAL transcripts only
        self.last_partial = ""  # Track last partial to avoid duplicates
        self.last_spoken_time = None
        self.last_sent_text = ""  # Track what was sent to UI
        self.last_send_time = 0  # For throttling
        self.silence_timeout = Config.SILENCE_TIMEOUT
        self.send_throttle = 0.3  # 300ms minimum between updates

    async def handle_transcript_event(self, event: TranscriptEvent):
        loop_time = asyncio.get_event_loop().time()

        for result in event.transcript.results:
            if not result.alternatives:
                continue

            text = (result.alternatives[0].transcript or "").strip()
            if not text:
                continue

            # Mark speech activity
            self.last_spoken_time = loop_time

            if result.is_partial:
                # ✅ Skip if identical to last partial
                if text == self.last_partial:
                    continue
                
                self.last_partial = text
                
                # Combine permanent buffer + current partial for display
                display_text = f"{self.buffer} {text}".strip() if self.buffer else text
                
                # ✅ Throttle updates: only send if enough time passed OR significant text change
                time_since_last = loop_time - self.last_send_time
                text_changed = len(display_text) - len(self.last_sent_text) > 5  # 5+ new chars
                
                if text_changed or time_since_last > self.send_throttle:
                    self.last_sent_text = display_text
                    self.last_send_time = loop_time
                    
                    await self.ws.send_json({
                        "type": "candidate_live", 
                        "text": display_text
                    })
            else:
                # ✅ Final result - add to permanent buffer
                if text:
                    self.buffer = f"{self.buffer} {text}".strip() if self.buffer else text
                    self.last_partial = ""  # Reset partial tracker
                    self.last_sent_text = self.buffer
                    self.last_send_time = loop_time
                    
                    await self.ws.send_json({
                        "type": "candidate_live", 
                        "text": self.buffer
                    })

    async def wait_for_final(self):
        """Wait until user stops speaking or user gives no response."""
        
        start_time = asyncio.get_event_loop().time()

        while True:
            await asyncio.sleep(0.5)
            now = asyncio.get_event_loop().time()

            # --- CASE 1: User NEVER speaks ---
            if not self.last_spoken_time:
                if now - start_time >= Config.INITIAL_WAIT_LIMIT:
                    final = self.buffer.strip()
                    self._reset_state()
                    return final or ""
                continue

            # --- CASE 2: User STOPPED speaking ---
            if now - self.last_spoken_time >= Config.SILENCE_TIMEOUT:
                final = self.buffer.strip()
                self._reset_state()
                return final

            # --- CASE 3: Max recording window hit ---
            if now - start_time >= Config.SPEAKING_WINDOW:
                final = self.buffer.strip()
                self._reset_state()
                return final

    def _reset_state(self):
        """Helper to reset all state variables"""
        self.buffer = ""
        self.last_partial = ""
        self.last_spoken_time = None
        self.last_sent_text = ""
        self.last_send_time = 0

    def set_question_number(self, qn: int):
        """Reset buffer for new question."""
        self.current_question = qn
        self._reset_state()

    async def pop_final_paragraph(self) -> str:
        """Retrieve and clear final paragraph"""
        final = self.buffer
        self._reset_state()
        return final


# ================== FACE MONITORING ==================
async def monitor_faces(
    session_id: str,
    email: str,
    frame_queue: asyncio.Queue,
    safe_send: callable,
    check_interval: float = Config.FACE_CHECK_INTERVAL
):
    """Monitor face detection in video frames"""
    mp_face_detection = mp.solutions.face_detection
    tracker = get_violation_tracker(session_id)
    last_alert: Optional[str] = None
    last_frame_time = 0
    
    with mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    ) as face_detection:
        while True:
            try:
                try:
                    frame_packet = await asyncio.wait_for(
                        frame_queue.get(),
                        timeout=check_interval
                    )
                except asyncio.TimeoutError:
                    if last_alert:
                        await safe_send({
                            "type": "popup",
                            "text": last_alert,
                            "level": "warning"
                        })
                    continue
                
                now = asyncio.get_event_loop().time()
                if now - last_frame_time < check_interval:
                    continue
                last_frame_time = now
                
                # Decode frame
                frame_bytes = base64.b64decode(frame_packet)
                np_arr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                
                # Process frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)
                num_faces = len(results.detections) if results.detections else 0
                
                # Determine alert and count violation
                violation_occurred = False
                if num_faces == 0:
                    popup_msg = "⚠️ No face detected!"
                    violation_occurred = True
                elif num_faces > 1:
                    popup_msg = "⚠️ Multiple faces detected!"
                    violation_occurred = True
                else:
                    popup_msg = "✓ Face detection normal"
                
                # Increment violation with cooldown
                if violation_occurred:
                    if now - tracker['last_violation_time'] >= tracker['cooldown']:
                        current_count = increment_violation(session_id)
                        
                        await safe_send({
                            "type": "popup",
                            "text": f"{popup_msg} (Violation #{current_count})",
                            "level": "warning",
                            "violation_count": current_count
                        })
                    else:
                        # Still show warning but don't count
                        await safe_send({
                            "type": "popup",
                            "text": popup_msg,
                            "level": "warning",
                            "violation_count": tracker['count']
                        })
                else:
                    await safe_send({
                        "type": "popup",
                        "text": popup_msg,
                        "level": "info",
                        "violation_count": tracker['count']
                    })
                
                last_alert = popup_msg
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Face monitoring error: {e}")

# ================== INTERVIEW ORCHESTRATOR ==================
@dataclass
class InterviewState:
    """Interview session state"""
    ws: WebSocket
    session_id: str
    email: str
    skills: str
    mode: str
    ws_lock: asyncio.Lock
    frame_queue: asyncio.Queue
    audio_queue: Optional[asyncio.Queue] = None
    transcribe_stream = None
    transcribe_active: bool = False
    transcript_handler: Optional[TranscriptionHandler] = None
    current_interaction_id: Optional[str] = None
    question_count: int = 1
    conversation: List[Dict] = None
    handler_task: Optional[asyncio.Task] = None
    mic_task: Optional[asyncio.Task] = None
    alert_task: Optional[asyncio.Task] = None
    auto_task: Optional[asyncio.Task] = None
    
    def __post_init__(self):
        if self.conversation is None:
            self.conversation = []

async def safe_send(state: InterviewState, msg: dict):
    """Thread-safe WebSocket send"""
    async with state.ws_lock:
        await state.ws.send_json(msg)


async def send_agent_message(state: InterviewState, text: str):
    """Send agent question + handle transcription pause"""
    
    # Stop recording before Polly speaks
    await stop_transcription(state)

    # Skip Polly voice for coding questions
    if state.question_count in Config.CODING_QUESTION_NUMBERS:
        await safe_send(state, {"type": "agent", "text": text, "audio": ""})
    else:
        audio_bytes = text_to_speech(text)
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        await safe_send(state, {"type": "agent", "text": text, "audio": audio_b64})

    # Wait before enabling transcription
    await asyncio.sleep(Config.INITIAL_WAIT_LIMIT)  # Default: 7 seconds

    # Restart transcription (user now allowed to speak)
    await start_transcription(state)


async def start_transcription(state: InterviewState):
    """Start AWS Transcribe streaming"""
    if state.transcribe_active:
        return
    
    try:
        client = TranscribeStreamingClient(region=Config.AWS_REGION)
        state.transcribe_stream = await client.start_stream_transcription(
            language_code="en-US",
            media_sample_rate_hz=Config.SAMPLE_RATE,
            media_encoding="pcm",
        )
        
        state.audio_queue = asyncio.Queue()
        state.transcript_handler = TranscriptionHandler(
            state.transcribe_stream.output_stream,
            state.ws
        )
        
        state.handler_task = asyncio.create_task(
            state.transcript_handler.handle_events()
        )
        state.mic_task = asyncio.create_task(
            stream_audio_to_transcribe(
                state.transcribe_stream.input_stream,
                state.audio_queue
            )
        )
        
        state.transcribe_active = True
        await safe_send(state, {"type": "status", "message": "Recording started"})
    except Exception as e:
        logger.error(f"Failed to start transcription: {e}")
        raise

async def stop_transcription(state: InterviewState):
    """Stop AWS Transcribe streaming safely (avoid double-close errors)."""
    if not state.transcribe_active:
        return

    try:
        # Prevent double-stop
        state.transcribe_active = False

        if state.audio_queue:
            await state.audio_queue.put(None)

        if state.mic_task:
            state.mic_task.cancel()

        if state.handler_task:
            state.handler_task.cancel()

        # Only attempt stream close if it's still open
        if state.transcribe_stream and state.transcribe_stream.input_stream:
            with contextlib.suppress(Exception):
                await state.transcribe_stream.input_stream.end_stream()

    except Exception as e:
        logger.warning(f"[safe-stop] Transcribe already closed: {e}")

    finally:
        state.transcribe_stream = None
        state.transcript_handler = None




async def send_next_question(state: InterviewState):
    """Generate and send next interview question"""
    if state.question_count > Config.MAX_QUESTIONS:
        await end_interview(state)
        return
    
    # Generate question
    if state.question_count in Config.CODING_QUESTION_NUMBERS:
        # ✅ Pass question_number to generate different coding questions
        question = await generate_coding_question(
            state.skills, 
            state.mode,
            state.question_count  # ADD THIS
        )
    else:
        question = await generate_non_coding_question(
            state.conversation,
            state.skills,
            state.question_count,
            state.mode
        )
    
    if not question.strip():
        raise RuntimeError("Failed to generate question")
    
    if state.transcript_handler:
        state.transcript_handler.set_question_number(state.question_count)
    
    state.conversation.append({"role": "assistant", "content": question})
    
    state.current_interaction_id = await save_interaction(
        state.email,
        question,
        candidate_text="",
        question_number=state.question_count,
        session_id=state.session_id,
        skill=state.skills,
        mode=state.mode
    )
    
    await send_agent_message(state, question)


async def wait_for_code_submission(state):
    """Poll DB until candidate submits code or timeout expires"""
    start = asyncio.get_event_loop().time()

    while True:
        if state.current_interaction_id:
            row = await execute_query(
                "SELECT candidate_speak FROM interaction_table WHERE id = %s",
                (state.current_interaction_id,),
                fetch=True
            )
            if row and row.get("candidate_speak"):
                return row["candidate_speak"], True
        
        # Timeout
        if asyncio.get_event_loop().time() - start >= Config.CODING_TIME_LIMIT:
            return "", False

        await asyncio.sleep(0.5)




async def collect_candidate_response(state: InterviewState) -> tuple[str, bool]:
    """Collect candidate response depending on question type"""
    
    is_coding = state.question_count in Config.CODING_QUESTION_NUMBERS

    # ------------------- CODING MODE -------------------
    if is_coding:
        return await wait_for_code_submission(state)

    # ------------------- SPEECH MODE -------------------
    # Start listening AFTER Polly delay handled earlier
    await start_transcription(state)

    # Wait until user stops speaking based on silence detection
    answer = await state.transcript_handler.wait_for_final()
    answered = bool(answer.strip())

    # Stop taking input once speech ends
    await stop_transcription(state)

    return answer.strip(), answered

    

async def process_question_loop(state: InterviewState):
    """Main loop to process all interview questions"""
    while state.question_count <= Config.MAX_QUESTIONS:
        # Collect response
        answer_text, answered = await collect_candidate_response(state)
        
        # Save response
        if answered and state.current_interaction_id:
            await update_candidate_response(state.current_interaction_id, answer_text)
            await safe_send(state, {"type": "candidate_final", "text": answer_text})
            state.conversation.append({"role": "user", "content": answer_text})
        else:
            if state.current_interaction_id:
                await update_candidate_response(state.current_interaction_id, "[no response]")
            state.conversation.append({"role": "user", "content": "[no response]"})
            await safe_send(state, {"type": "candidate_final", "text": "[no response]"})
        
        # Move to next question
        state.question_count += 1
        if state.question_count <= Config.MAX_QUESTIONS:
            await send_next_question(state)
    
    await end_interview(state)

async def evaluate_interview(state: InterviewState) -> Dict:
    """Evaluate interview and return results with penalties"""
    interactions = await get_session_interactions(state.session_id, state.email)
    
    if not interactions:
        raise ValueError("No interactions found for evaluation")
    
    total_score = 0.0
    answered_count = 0
    
    for inter in interactions:
        agent_text = inter.get('agent_speak', '') or ''
        candidate_text = inter.get('candidate_speak', '') or ''
        
        if candidate_text.strip() and candidate_text not in ["[no response]", ""]:
            answered_count += 1
        
        score = await score_response(agent_text, candidate_text)
        total_score += score
    
    # Get violation count from memory
    violation_count = get_violation_count_memory(state.session_id)
    
    # Apply penalty: -1 point per violation
    penalty = violation_count * 1.0  # 1 point per violation
    total_score = max(0.0, total_score - penalty)
    total_score = min(total_score, Config.MAX_SCORE)
    
    logger.info(
        f"Session {state.session_id} - Raw score: {total_score + penalty:.2f}, "
        f"Violations: {violation_count}, Penalty: {penalty:.2f} points, "
        f"Final score: {total_score:.2f}"
    )
    
    # Generate comprehensive feedback
    feedback = await generate_feedback(
        interactions,
        total_score,
        answered_count,
        violation_count,
        penalty,
        state.skills
    )
    
    # Get student data
    student = await get_student(state.email)
    reg_number = student['reg_number'] if student else None
    
    # Save score with violation count
    await save_interview_score(
        session_id=state.session_id,
        reg_number=reg_number,
        email=state.email,
        score=total_score,
        max_score=Config.MAX_SCORE,
        questions_asked=Config.MAX_QUESTIONS,
        answered=answered_count,
        relevance_feedback=feedback["relevance_feedback"],
        focus_feedback=feedback["focus_feedback"],
        overall_summary=feedback["overall_summary"],
        skill=state.skills,
        communication_strength=feedback["communication_strength"],
        technical_strength=feedback["technical_strength"],
        mode=state.mode,
        violation_count=violation_count
    )
    
    # Clear tracker after saving
    clear_violation_tracker(state.session_id)
    
    return {
        "session_id": state.session_id,
        "email": state.email,
        "score": round(total_score, 2),
        "max_score": Config.MAX_SCORE,
        "questions_asked": Config.MAX_QUESTIONS,
        "answered": answered_count,
        "relevance_feedback": feedback["relevance_feedback"],
        "focus_feedback": feedback["focus_feedback"],
        "overall_summary": feedback["overall_summary"],
        "communication_strength": feedback["communication_strength"],
        "technical_strength": feedback["technical_strength"],
        "skill": state.skills,
        "reg_number": reg_number,
        "mode": state.mode,
        "violation_count": violation_count,
        "penalty_applied": penalty
    }

async def end_interview(state: InterviewState):
    """End interview and send evaluation"""
    closing = "Thank you for your time today. You can leave the interview now."
    state.conversation.append({"role": "assistant", "content": closing})
    await send_agent_message(state, closing)
    await stop_transcription(state)
    
    evaluation_result = await evaluate_interview(state)
    await safe_send(state, {"type": "evaluation_done", "result": evaluation_result})
    
    try:
        await state.ws.close()
    except Exception:
        pass

async def start_interview(state: InterviewState):
    """Start the interview"""
    state.alert_task = asyncio.create_task(
        monitor_faces(
            state.session_id,
            state.email,
            state.frame_queue,
            lambda msg: safe_send(state, msg)
        )
    )
    
    state.conversation.append({
        "role": "user",
        "content": f"Interview started with skills: {state.skills}, mode: {state.mode}"
    })
    
    await send_next_question(state)
    state.auto_task = asyncio.create_task(process_question_loop(state))

async def cleanup_interview(state: InterviewState):
    """Cleanup resources"""
    try:
        await stop_transcription(state)
    except Exception:
        pass
    
    for task in [state.auto_task, state.alert_task]:
        if task:
            try:
                task.cancel()
            except Exception:
                pass


# ================== API ENDPOINTS ==================
@app.post("/session/create", status_code=status.HTTP_201_CREATED)
async def create_session_endpoint(email: str = Form(...)):
    """Create a new interview session"""
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email is required"
        )
    
    session_id = await create_session(email)
    return {
        "session_id": session_id,
        "message": "Session created successfully"
    }

@app.websocket("/ws/interview")
async def interview_websocket(
    ws: WebSocket,
    session_id: str = Query(...),
    email: str = Query(...),
    skill: str = Query(...),
    mode: str = Query(...)
):
    """WebSocket endpoint for conducting interviews"""
    await ws.accept()
    
    state = None
    
    try:
        # Validate session
        await validate_session(session_id, email)
        
        # Deduct token
        new_balance = await deduct_token(email)
        await ws.send_json({
            "type": "token_update",
            "message": "Token deducted for interview",
            "tokens_remaining": new_balance
        })
        
        # Initialize state
        state = InterviewState(
            ws=ws,
            session_id=session_id,
            email=email,
            skills=skill,
            mode=mode,
            ws_lock=asyncio.Lock(),
            frame_queue=asyncio.Queue()
        )
        
        await start_interview(state)
        
        # Handle incoming messages
        while True:
            packet = await ws.receive()
            
            if packet["type"] == "websocket.disconnect":
                break
            
            # Handle binary audio data
            if "bytes" in packet and packet["bytes"] is not None:
                if state.transcribe_active and state.audio_queue:
                    await state.audio_queue.put(packet["bytes"])
                continue
            
            # Handle text messages
            if "text" in packet and packet["text"]:
                data = json.loads(packet["text"])
                
                # Frame data for face monitoring
                if data.get("type") == "frame" and "data" in data:
                    await state.frame_queue.put(data["data"])
                
                # Code execution
                elif data.get("type") == "run_code":
                    code = data.get("code", "")
                    result = await run_code(code)
                    await ws.send_json({"type": "run_result", "output": result})
                
                # Code submission
                elif data.get("type") == "submit_code":
                    code = data.get("code", "")
                    if state.current_interaction_id:
                        await update_candidate_response(state.current_interaction_id, code)
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except HTTPException as e:
        await ws.send_json({"type": "error", "message": e.detail})
        try:
            await ws.close()
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket: {e}")
        try:
            await ws.send_json({"type": "error", "message": "Internal server error"})
            await ws.close()
        except Exception:
            pass
    finally:
        if state:
            await cleanup_interview(state)

@app.get("/session/{session_id}/violations")
async def get_session_violations_endpoint(session_id: str):
    """Get violation statistics for a session"""
    violation_count = get_violation_count_memory(session_id)
    penalty = violation_count * 1.0  # 1 point per violation
    
    return {
        "session_id": session_id,
        "total_violations": violation_count,
        "penalty_applied": penalty,
        "penalty_per_violation": 1.0
    }



# ================== MAIN ==================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "master:app",
        host="0.0.0.0",
        port=8006,
        log_level="info",
        access_log=True
    )