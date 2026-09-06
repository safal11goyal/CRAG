"""PostgreSQL Database Module for RAG Conversation Storage.

Provides SQLAlchemy models, table initialization, and logging functions for
storing conversation sessions, user queries, pipeline rewrites, retrieved sources,
generated answers, verification flags, and confidence scores in PostgreSQL.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    desc,
    text,
)
from sqlalchemy.orm import declarative_base, relationship, scoped_session, sessionmaker

import config

logger = logging.getLogger("RAGDatabase")

Base = declarative_base()

_engine = None
_SessionFactory = None
_is_connected = False


# ============================================================================
# ORM MODELS
# ============================================================================

class Conversation(Base):
    """Represents a chat session."""
    __tablename__ = "conversations"

    session_id = Column(String(64), primary_key=True, index=True)
    title = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    turns = relationship(
        "ConversationTurn",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ConversationTurn.created_at",
    )


class ConversationTurn(Base):
    """Represents an individual question-answer turn and RAG pipeline diagnostics."""
    __tablename__ = "conversation_turns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        String(64),
        ForeignKey("conversations.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_query = Column(Text, nullable=False)
    search_query = Column(Text, nullable=True)
    answer = Column(Text, nullable=False)
    sources = Column(JSON, nullable=True)
    enough_evidence = Column(Boolean, default=False)
    verification_reason = Column(Text, nullable=True)
    confidence = Column(Float, default=0.0)
    warning = Column(Text, nullable=True)
    iterations = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    conversation = relationship("Conversation", back_populates="turns")


# ============================================================================
# DATABASE INITIALIZATION & CONNECTIVITY
# ============================================================================

def init_db(database_url: Optional[str] = None) -> bool:
    """Initialize PostgreSQL database engine and create tables if they do not exist.

    Args:
        database_url: Optional override connection string. Defaults to config.get_database_url().

    Returns:
        True if connected and tables are verified, False otherwise.
    """
    global _engine, _SessionFactory, _is_connected

    url = database_url or config.get_database_url()

    if not url:
        logger.warning("No PostgreSQL DATABASE_URL configured. Running in offline/in-memory mode.")
        _is_connected = False
        return False

    try:
        connect_args = {}
        if "postgres" in url.lower():
            connect_args["connect_timeout"] = 5

        # Configure connection pool with connection timeout
        _engine = create_engine(
            url,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args=connect_args,
        )

        # Test connectivity
        with _engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        # Create tables
        Base.metadata.create_all(bind=_engine)

        _SessionFactory = scoped_session(
            sessionmaker(autocommit=False, autoflush=False, bind=_engine)
        )
        _is_connected = True
        logger.info("Successfully connected to PostgreSQL database and verified tables.")
        return True

    except Exception as exc:
        logger.warning(
            "Could not connect to PostgreSQL at '%s': %s. Conversation persistence will run in offline mode.",
            url.split("@")[-1] if "@" in url else url,
            exc,
        )
        _is_connected = False
        return False


def is_db_connected() -> bool:
    """Return True if PostgreSQL database is connected and active."""
    return _is_connected


def close_db() -> None:
    """Safely close and dispose database engine connections."""
    global _engine, _SessionFactory, _is_connected
    if _SessionFactory is not None:
        _SessionFactory.remove()
    if _engine is not None:
        _engine.dispose()
    _is_connected = False


# ============================================================================
# CONVERSATION OPERATIONS
# ============================================================================

def create_conversation(
    session_id: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    """Register a new conversation session in PostgreSQL.

    Args:
        session_id: Optional unique identifier. Generates UUID4 string if None.
        title: Optional title for the conversation.

    Returns:
        The session_id string.
    """
    sid = session_id or str(uuid.uuid4())

    if not _is_connected or _SessionFactory is None:
        return sid

    session = _SessionFactory()
    try:
        existing = session.query(Conversation).filter_by(session_id=sid).first()
        if not existing:
            conv = Conversation(session_id=sid, title=title or f"Session {sid[:8]}")
            session.add(conv)
            session.commit()
    except Exception as exc:
        logger.error("Failed to create conversation in PostgreSQL: %s", exc)
        session.rollback()
    finally:
        session.close()

    return sid


def save_conversation_turn(
    session_id: str,
    user_query: str,
    search_query: str,
    answer: str,
    sources: Optional[List[str]] = None,
    enough_evidence: bool = False,
    verification_reason: str = "",
    confidence: float = 0.0,
    warning: str = "",
    iterations: int = 0,
) -> bool:
    """Save an individual query/response turn with RAG metadata to PostgreSQL.

    Args:
        session_id: Unique conversation identifier.
        user_query: Original question asked by the user.
        search_query: Reformulated search query used by the pipeline.
        answer: Final answer generated by the LLM.
        sources: List of source documents or URLs cited.
        enough_evidence: Verification flag indicating if evidence was sufficient.
        verification_reason: Verification analysis reasoning.
        confidence: Estimated answer confidence score (0.0 to 1.0).
        warning: Any hallucination or fallback notices.
        iterations: Number of retrieval/reformulation cycles performed.

    Returns:
        True if persisted successfully, False otherwise.
    """
    if not _is_connected or _SessionFactory is None:
        return False

    session = _SessionFactory()
    try:
        # Ensure parent conversation exists
        conv = session.query(Conversation).filter_by(session_id=session_id).first()
        if not conv:
            # Generate a title from the first 50 characters of user query
            title = (user_query[:50] + "...") if len(user_query) > 50 else user_query
            conv = Conversation(session_id=session_id, title=title)
            session.add(conv)
            session.flush()
        else:
            conv.updated_at = datetime.now(timezone.utc)

        turn = ConversationTurn(
            session_id=session_id,
            user_query=user_query,
            search_query=search_query,
            answer=answer,
            sources=sources or [],
            enough_evidence=enough_evidence,
            verification_reason=verification_reason,
            confidence=float(confidence),
            warning=warning,
            iterations=int(iterations),
        )
        session.add(turn)
        session.commit()
        return True

    except Exception as exc:
        logger.error("Failed to save conversation turn to PostgreSQL: %s", exc)
        session.rollback()
        return False
    finally:
        session.close()


def get_conversation_history(session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve previous conversation turns for a given session.

    Args:
        session_id: The conversation session identifier.
        limit: Maximum number of recent turns to retrieve.

    Returns:
        List of turn dictionaries ordered chronologically.
    """
    if not _is_connected or _SessionFactory is None:
        return []

    session = _SessionFactory()
    try:
        turns = (
            session.query(ConversationTurn)
            .filter_by(session_id=session_id)
            .order_by(ConversationTurn.created_at.asc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": t.id,
                "user_query": t.user_query,
                "search_query": t.search_query,
                "answer": t.answer,
                "sources": t.sources or [],
                "enough_evidence": t.enough_evidence,
                "confidence": t.confidence,
                "warning": t.warning,
                "iterations": t.iterations,
                "created_at": t.created_at.isoformat() if t.created_at else None,
            }
            for t in turns
        ]
    except Exception as exc:
        logger.error("Failed to load conversation history for session '%s': %s", session_id, exc)
        return []
    finally:
        session.close()


def list_conversations(limit: int = 10) -> List[Dict[str, Any]]:
    """List recent conversation sessions with metadata.

    Args:
        limit: Maximum number of sessions to return.

    Returns:
        List of conversation summary dictionaries.
    """
    if not _is_connected or _SessionFactory is None:
        return []

    session = _SessionFactory()
    try:
        conversations = (
            session.query(Conversation)
            .order_by(desc(Conversation.updated_at))
            .limit(limit)
            .all()
        )
        results = []
        for c in conversations:
            turn_count = len(c.turns) if c.turns else 0
            results.append({
                "session_id": c.session_id,
                "title": c.title or "Untitled",
                "turns_count": turn_count,
                "created_at": c.created_at.isoformat() if c.created_at else None,
                "updated_at": c.updated_at.isoformat() if c.updated_at else None,
            })
        return results
    except Exception as exc:
        logger.error("Failed to list conversations from PostgreSQL: %s", exc)
        return []
    finally:
        session.close()
