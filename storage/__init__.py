"""Database configuration and initialization for parry agent."""

from storage.engine import engine, init_database
from storage.models import (
    Conversation,
    Message,
    MessageLog,
    ConversationLog,
)

# Initialize database on import
init_database()

__all__ = [
    "Conversation",
    "ConversationLog",
    "Message",
    "MessageLog",
    "engine",
    "init_database",
]
