from app import db
from app.models.base_model import BaseModel
from sqlalchemy import Column, TIMESTAMP, VARCHAR, BOOLEAN

class User(BaseModel):
    user_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(200), nullable=True)
    email = db.Column(db.String(200), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False, nullable=False)
    latest_login = db.Column(TIMESTAMP, nullable=True)