from fastapi import Header , HTTPException
from fastapi import Request
from core.sessions import Session

MAX_SESSION_ID_LENGTH = 64

def get_session_id(x_session_id:str = Header(default=None))-> str:
    if not x_session_id or not x_session_id.strip():
        raise HTTPException(status_code=400, detail="Missing X-Session-Id header")
    session_id = x_session_id.strip()
    if len(session_id)>MAX_SESSION_ID_LENGTH:
        raise HTTPException(status_code=400, detail="Invalid X-Session-ID header")
    return session_id

def get_session(request: Request, x_session_id: str | None = Header(default=None)) -> Session:
    if not x_session_id or len(x_session_id) > MAX_SESSION_ID_LENGTH:
        raise HTTPException(status_code=401, detail="Missing or invalid session")
    session = request.app.state.sessions.get(x_session_id.strip())
    if session is None:
        raise HTTPException(status_code=401, detail="Session not found or expired")
    return session