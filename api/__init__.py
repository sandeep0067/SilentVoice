"""
SilentVoice API package.
"""

try:
    from api.app import app
    __all__ = ['app']
except ImportError:
    # FastAPI not installed
    __all__ = []
