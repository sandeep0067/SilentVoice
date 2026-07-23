from fastapi import Depends


def get_settings():
    from app.core.config import settings
    return settings
