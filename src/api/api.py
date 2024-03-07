from typing import List, Optional, Tuple, Union

from fastapi import Depends, FastAPI, File, Header, Request, UploadFile, APIRouter
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api import chat, collections, users
from src.config import config, get_vector_db
from src.schema import IngestItem


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "*"  # React app
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a router with a prefix
router = APIRouter(prefix="/api")

# Create a local session factory
engine = create_engine(config.sql_connection_str, echo=config.verbose)
LocalSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)

config.print()


def get_db():
    db_session = None
    try:
        db_session = LocalSession()
        yield db_session
    finally:
        if db_session:
            db_session.close()


class AuthInfo(BaseModel):
    username: str
    token: str
    roles: List[str] = []


# placeholder for extracting the Auth info from the request
async def get_auth_user(
    request: Request, x_username: Union[str, None] = Header(None)
) -> AuthInfo:
    """Get the user from the database"""
    token = request.cookies.get("Authorization", "")
    if x_username:
        return AuthInfo(username=x_username, token=token)
    else:
        return AuthInfo(username="yhaviv@gmail.com", token=token)


@router.post("/query")
async def query(
    item: chat.QueryItem, session=Depends(get_db), auth=Depends(get_auth_user)
):
    """This is the query command"""
    x = chat.query(session, item, username=auth.username)
    return x


@router.get("/collections")
async def list_collections(
    owner: str = None,
    metadata: Optional[List[Tuple[str, str]]] = None,
    names_only: bool = True,
    session=Depends(get_db),
):
    return collections.list_collections(
        session, owner=owner, metadata=metadata, names_only=names_only
    )


@router.get("/collection/{name}")
async def get_collection(name: str, short: bool = False, session=Depends(get_db)):
    return collections.get_collection(session, name, short=short)


@router.post("/collection/{name}")
async def create_collection(
    request: Request,
    name: str,
    session=Depends(get_db),
    auth: AuthInfo = Depends(get_auth_user),
):
    data = await request.json()
    return collections.create_collection(
        session, name, owner_name=auth.username, **data
    )


@router.post("/collection/{name}/ingest")
async def ingest(name, item: IngestItem, session=Depends(get_db)):
    return collections.ingest(session, name, item)


@router.get("/users")
async def list_users(
    email: str = None,
    username: str = None,
    names_only: bool = True,
    short: bool = False,
    session=Depends(get_db),
):
    return users.get_users(
        session, email=email, username=username, names_only=names_only, short=short
    )


@router.get("/user/{username}")
async def get_user(username: str, session=Depends(get_db)):
    return users.get_user(session, username)


@router.post("/user/{username}")
async def create_user(
    request: Request,
    username: str,
    session=Depends(get_db),
):
    """This is the user command"""
    data = await request.json()
    return users.create_user(session, username, **data)


@router.delete("/user/{username}")
async def delete_user(username: str, session=Depends(get_db)):
    return users.delete_user(session, username)


# add routs for chat sessions, list_sessions, get_session
@router.get("/sessions")
async def list_sessions(
    user: str = None,
    last: int = 0,
    created: str = None,
    short: bool = False,
    session=Depends(get_db),
):
    return chat.list_sessions(
        session, user, created_after=created, last=last, short=short
    )


@router.get("/session/{session_id}")
async def get_session(session_id: str, session=Depends(get_db)):
    return chat.get_session(session, session_id)


@router.post("/transcribe")
async def transcribe_file(file: UploadFile = File(...)):
    file_contents = await file.read()
    file_handler = file.file
    return chat.transcribe_file(file_handler)


@router.get("/tst")
async def tst():
    vector = get_vector_db(config)
    results = vector.similarity_search(
        "Can you please provide me with information about the mobile plans?"
    )
    print(results)

# Include the router in the main app
app.include_router(router)