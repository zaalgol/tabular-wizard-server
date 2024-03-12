from sqlalchemy import Column, TIMESTAMP, VARCHAR, BOOLEAN
from app import db

class BaseModel(db.Model):
    __abstract__ = True  
    created = db.Column('deleted', TIMESTAMP)
    updated = db.Column('deleted', TIMESTAMP)
    isDeleted = db.Column('isDeleted', db.BOOLEAN)
    