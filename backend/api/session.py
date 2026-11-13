from fastapi import Header , HTTPException

MAX_SESSION_ID_LENGTH = 64

def get_session_id(x_session_id:str = Header(default=None))-> str:
    if not x_session_id or not x_session_id.strip():
        raise HTTPException(status_code=400, detail="Missing X-Session-Id header")
    session_id = x_session_id.strip()
    if len(session_id)>MAX_SESSION_ID_LENGTH:
        raise HTTPException(status_code=400, detail="Invalid X-Session-ID header")
    return session_id

    