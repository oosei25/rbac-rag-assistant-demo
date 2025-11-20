from enum import Enum
from pydantic import BaseModel

class Role(str, Enum):
    finance="finance"
    marketing="marketing"
    hr="hr"
    engineering="engineering"
    employee="employee"
    clevel="clevel"

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: Role

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]  # file paths
