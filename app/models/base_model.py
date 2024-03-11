from sqlalchemy import Column, TIMESTAMP, VARCHAR, BOOLEAN
from app import db

class BaseModel(db.Model):
    created = db.Column('deleted', TIMESTAMP)
    updated = db.Column('deleted', TIMESTAMP)
    isDeleted = db.Column('isDeleted', TIMESTAMP)
    