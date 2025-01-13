"""Session-specific exceptions."""

from __future__ import annotations

import httpx


class ChatSessionError(Exception):
    """Base exception for chat session-related errors."""


class ChatSessionNotFoundError(ChatSessionError):
    """Raised when trying to access a non-existent session."""


class ChatSessionConfigError(ChatSessionError):
    """Raised when chat session configuration is invalid."""


def format_error(error: Exception) -> str:
    """Format error message for display."""
    # Known error types we want to handle specially
    match error:
        case ChatSessionConfigError():
            return f"Chat session error: {error}"
        case ValueError() if "token" in str(error):
            return "Connection interrupted"
        case httpx.ReadError():
            return "Connection lost. Please try again."
        case GeneratorExit():
            return "Response stream interrupted"
        case _:
            return f"Error: {error}"
