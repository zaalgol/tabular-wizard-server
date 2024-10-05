import json
from datetime import datetime, timedelta
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi import WebSocket
from app.entities.model import Model
from app.services.model_service import ModelService
from app.services.token_service import TokenService
from app.services.user_service import UserService
from pymongo.database import Database


router = APIRouter()

def get_db(request: Request) -> Database:
    return request.app.state.db

def get_user_service(db: Database = Depends(get_db)) -> UserService:
    return UserService(db)

def get_token_service() -> TokenService:
    return TokenService()

def get_model_service(db: Database = Depends(get_db)) -> ModelService:
    return ModelService(db)

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
    return user_service.login(email, password)

# @router.post('/api/trainModel/', status_code=status.HTTP_200_OK)
# async def train_model(
#     request: Request, 
#     background_tasks: BackgroundTasks,
#     user_id: str = Depends(TokenService.extract_user_id_from_token),
#     model_service: ModelService = Depends(get_model_service),
#     token_service: TokenService = Depends(get_token_service),
#     user_service: UserService = Depends(get_user_service),
# ):
#     data = await request.json()
#     user = user_service.get_user_by_id(user_id)
#     if not user:
#         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    
#     model = Model(
#         user_id=user_id,
#         file_name=data.get('fileName'),
#         model_name=data.get('modelName'),
#         description=data.get('description'),
#         model_type=data.get('modelType'),
#         training_strategy=data.get('trainingStrategy'),
#         sampling_strategy=data.get('samplingStrategy'),
#         target_column=data.get('targetColumn'),
#         metric=data.get('metric'),
#         is_time_series=data.get('isTimeSeries', False)
#     )

#     return model_service.train_model(model, data.get('dataset'), background_tasks)

@router.post('/api/trainModel/', status_code=status.HTTP_200_OK)
async def train_model(
    request: Request,
    user_id: str = Depends(TokenService.extract_user_id_from_token),
    model_service: ModelService = Depends(get_model_service),
    user_service: UserService = Depends(get_user_service),
):
    data = await request.json()
    user = user_service.get_user_by_id(user_id)
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
        is_time_series=data.get('isTimeSeries', False)
    )

    # Remove background_tasks
    result = await model_service.train_model(model, data.get('dataset'))

    return JSONResponse(content=result, status_code=200)

@router.get('/api/userModels/', status_code=status.HTTP_200_OK)
async def get_user_models(
    user_id: str = Depends(TokenService.extract_user_id_from_token),
    model_service: ModelService = Depends(get_model_service),
    user_service: UserService = Depends(get_user_service)
):
    user = user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    models = model_service.get_user_models_by_id(user_id)

    json_compatible_models = json.loads(json.dumps({"models": models}, cls=DateTimeEncoder))
    return JSONResponse(content=json_compatible_models)

@router.get('/api/model', status_code=status.HTTP_200_OK)
async def get_user_model(
    model_name: str, 
    user_id: str = Depends(TokenService.extract_user_id_from_token),
    model_service: ModelService = Depends(get_model_service),
    token_service: TokenService = Depends(get_token_service),
    user_service: UserService = Depends(get_user_service)
):
    user = user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    model = model_service.get_user_model_by_user_id_and_model_name(user_id, model_name)
    
    json_compatible_models = json.loads(json.dumps({"model": model}, cls=DateTimeEncoder))
    return JSONResponse(content=json_compatible_models)

@router.get('/api/modelMetric', status_code=status.HTTP_200_OK)
async def get_model_evaluations(
    model_name: str, 
    user_id: str = Depends(TokenService.extract_user_id_from_token),
    model_service: ModelService = Depends(get_model_service),
    token_service: TokenService = Depends(get_token_service),
    user_service: UserService = Depends(get_user_service)
):
    user = user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    await model_service.get_model_details_file(user_id, model_name)
    return JSONResponse(content={}, status_code=status.HTTP_200_OK)

@router.delete('/api/model', status_code=status.HTTP_200_OK)
async def delete_model(
    model_name: str, 
    user_id: str = Depends(TokenService.extract_user_id_from_token),
    model_service: ModelService = Depends(get_model_service),
    token_service: TokenService = Depends(get_token_service),
    user_service: UserService = Depends(get_user_service)
):
    user = user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    result = model_service.delete_model_of_user(user_id, model_name)
    return JSONResponse(content={}, status_code=status.HTTP_200_OK)

@router.post('/api/inference/', status_code=status.HTTP_200_OK)
async def inference(
    request: Request,
    user_id: str = Depends(TokenService.extract_user_id_from_token),
    model_service: ModelService = Depends(get_model_service),
    user_service: UserService = Depends(get_user_service),
):
    data = await request.json()
    user = user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    
    dataset = data.get('dataset')
    model_name = data.get('modelName')
    file_name = data.get('fileName')

    result = await model_service.inference(user_id=user_id, model_name=model_name, file_name=file_name, dataset=dataset)

    return JSONResponse(content=result, status_code=200)

@router.get('/download/{filename}', status_code=status.HTTP_200_OK)
async def download_file(
    filename: str, 
    model_name: str, 
    file_type: str, 
    user_id: str = Depends(TokenService.extract_user_id_from_token),
    model_service: ModelService = Depends(get_model_service),
    user_service: UserService = Depends(get_user_service)
):
    user = user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    
    return model_service.download_file(user_id, model_name, filename, file_type)

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message text was: {data}")
