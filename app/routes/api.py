import json
from pymongo.database import Database
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Header, status, Request, WebSocket
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from fastapi import WebSocket
from app.entities.model import Model
from app.logger_setup import setup_logger
from app.services.model_service import ModelService
from app.services.token_service import TokenService
from app.services.user_service import UserService

logger = setup_logger(__name__)

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

def get_db(request: Request) -> Database:
    return request.app.state.db

def get_user_service(db: Database = Depends(get_db)) -> UserService:
    return UserService(db)

def get_token_service(db: Database = Depends(get_db)) -> TokenService:
    return TokenService(db)

def get_model_service(
    request: Request,
    db: Database = Depends(get_db)
) -> ModelService:
    return ModelService(request.app, db)

async def get_current_user_id(
    request: Request,
    authorization: str = Header(None),
    token_service: TokenService = Depends(get_token_service),
):
    token = None
    if authorization:
        # Extract token from header
        scheme, _, param = authorization.partition(' ')
        if scheme.lower() == 'bearer':
            token = param
    if not token:
        # Try to get token from query parameters
        token = request.query_params.get('Authorization')
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await token_service.extract_user_id_from_token(token)

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

@router.post('/api/login/', status_code=status.HTTP_200_OK)
async def login(request: Request, user_service: UserService = Depends(get_user_service)):
    data = await request.json()
    email = data.get('email')
    password = data.get('password')
    response = await user_service.login(email, password)
    if response.status_code == 200:
        logger.info(f"User {email} logged in successfully.")
    else:
        logger.warning(f"Failed login attempt for {email}.")
    return response

@router.post('/api/refresh_token/', status_code=status.HTTP_200_OK)
async def refresh_token(request: Request, token_service: TokenService = Depends(get_token_service)):
    refresh_token = request.cookies.get('refresh_token')
    if not refresh_token:
        logger.warning("Refresh token missing in cookies.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Refresh token missing")

    try:
        access_token, new_refresh_token = await token_service.refresh_access_token(refresh_token)
        response = JSONResponse({"access_token": access_token})
        response.set_cookie(key="refresh_token", value=new_refresh_token, httponly=True, secure=True)
        logger.info("Refresh token successfully rotated.")
        return response
    except HTTPException as e:
        logger.error(f"Invalid refresh token: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during token refresh: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server error")
    
@router.post("/api/register", status_code=status.HTTP_201_CREATED)
async def register(request: Request, user_service: UserService = Depends(get_user_service)):
    """
    Create a new user. If email already exists, return 400.
    """
    data = await request.json()
    email = data.get("email")
    password = data.get("password")
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required")

    # Check if user already exists
    existing_user = await user_service.user_repository.get_user_by_email(email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="User with this email already exists."
        )

    # Create the user
    created_user = await user_service.create_user(email, password)
    if not created_user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user."
        )

    return JSONResponse(content={"message": "User created successfully"}, status_code=201)

@router.post("/api/update_password", status_code=status.HTTP_200_OK)
async def update_password(
    request: Request,
    current_user_id: str = Depends(get_current_user_id),
    user_service: UserService = Depends(get_user_service)
):
    """
    Allow an authenticated user to update their password.
    """
    data = await request.json()
    new_password = data.get("new_password")
    if not new_password:
        raise HTTPException(status_code=400, detail="New password is required")

    # Update user password
    await user_service.update_user_password(current_user_id, new_password)

    return JSONResponse(content={"message": "Password updated successfully"}, status_code=200)

@router.post("/api/forgot_password", status_code=status.HTTP_200_OK)
async def forgot_password(request: Request, user_service: UserService = Depends(get_user_service)):
    """
    Generate a reset token for the user, 
    and presumably send it via email. 
    For demonstration, we'll just return it in JSON.
    """
    data = await request.json()
    email = data.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")
    
    reset_token = await user_service.request_password_reset(email)
    if not reset_token:
        # For security, you might still return 200 
        # to avoid revealing whether the email exists
        return JSONResponse({"message": "If that email exists, a reset was sent."})

    # In real usage, an email would be sent. We'll just return the token:
    return JSONResponse({"reset_token": reset_token})

@router.post("/api/reset_password", status_code=status.HTTP_200_OK)
async def confirm_reset_password(request: Request, user_service: UserService = Depends(get_user_service)):
    """
    Use the token from the forgot_password flow to set a new password.
    """
    data = await request.json()
    token = data.get("reset_token")
    new_password = data.get("new_password")
    if not token or not new_password:
        raise HTTPException(status_code=400, detail="reset_token and new_password are required.")

    try:
        await user_service.reset_password_with_token(token, new_password)
        return JSONResponse({"message": "Password has been reset successfully"})
    except HTTPException as e:
        # Pass along any token validation issues
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to reset password.")

@router.get('/api/userModels/', status_code=status.HTTP_200_OK)
async def get_user_models(
    request: Request,
    user_id: str = Depends(get_current_user_id),
    model_service: ModelService = Depends(get_model_service),
    user_service: UserService = Depends(get_user_service)
):
    user = await user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    models = await model_service.get_user_models_by_id(user_id)

    json_compatible_models = json.loads(json.dumps({"models": models}, cls=DateTimeEncoder))
    return JSONResponse(content=json_compatible_models)

@router.post('/api/trainModel/', status_code=status.HTTP_200_OK)
async def train_model(
    request: Request,
    user_id: str = Depends(get_current_user_id),
    model_service: ModelService = Depends(get_model_service),
    user_service: UserService = Depends(get_user_service),
):
    data = await request.json()
    user = await user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    
    model = Model(
        user_id=user_id,
        file_name=data.get('fileName'),
        model_name=data.get('modelName'),
        description=data.get('description'),
        model_type=data.get('modelType'),
        training_strategy=data.get('trainingStrategy'),
        sampling_strategy=data.get('samplingStrategy'),
        target_column=data.get('targetColumn'),
        metric=data.get('metric'),
        is_time_series=data.get('isTimeSeries', False),
        columns_type=data.get('columnsType')
    )

    result = await model_service.train_model(model, data.get('dataset'))

    return JSONResponse(content=result, status_code=200)

@router.get('/api/modelMetric', status_code=status.HTTP_200_OK)
async def get_model_evaluations(
    request: Request,
    model_name: str,
    user_id: str = Depends(get_current_user_id),
    model_service: ModelService = Depends(get_model_service),
    user_service: UserService = Depends(get_user_service)
):
    user = await user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    result = await model_service.get_model_details_file(user_id, model_name)
    return JSONResponse(content=result, status_code=status.HTTP_200_OK)

@router.get('/api/model', status_code=status.HTTP_200_OK)
async def get_user_model(
    request: Request,
    model_name: str,
    user_id: str = Depends(get_current_user_id),
    model_service: ModelService = Depends(get_model_service),
    user_service: UserService = Depends(get_user_service)
):
    user = await user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    model = await model_service.get_user_model_by_user_id_and_model_name(user_id, model_name)
    
    json_compatible_models = json.loads(json.dumps({"model": model}, cls=DateTimeEncoder))
    return JSONResponse(content=json_compatible_models)

@router.delete('/api/model', status_code=status.HTTP_200_OK)
async def delete_model(
    request: Request,
    model_name: str,
    user_id: str = Depends(get_current_user_id),
    model_service: ModelService = Depends(get_model_service),
    user_service: UserService = Depends(get_user_service)
):
    user = await user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    await model_service.delete_model_of_user(user_id, model_name)
    return JSONResponse(f"model {model_name} of user {user_id} has been successfully deleted", status_code=status.HTTP_200_OK)

@router.post('/api/inference/', status_code=status.HTTP_200_OK)
async def inference(
    request: Request,
    user_id: str = Depends(get_current_user_id),
    model_service: ModelService = Depends(get_model_service),
    user_service: UserService = Depends(get_user_service),
):
    data = await request.json()
    user = await user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    
    dataset = data.get('dataset')
    model_name = data.get('modelName')
    file_name = data.get('fileName')

    result = await model_service.inference(user_id=user_id, model_name=model_name, file_name=file_name, dataset=dataset)

    return JSONResponse(content=result, status_code=200)

@router.get('/download/{filename}', status_code=status.HTTP_200_OK)
async def download_file(
    request: Request,
    filename: str,
    model_name: str,
    file_type: str,
    user_id: str = Depends(get_current_user_id),
    model_service: ModelService = Depends(get_model_service),
    user_service: UserService = Depends(get_user_service)
):
    user = await user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    
    return model_service.download_file(user_id, model_name, filename, file_type)
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message text was: {data}")
