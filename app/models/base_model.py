from app import db
class BaseModel(db.Model):
    __abstract__ = True  
    created = db.Column(db.TIMESTAMP)
    updated = db.Column(db.TIMESTAMP)
    isDeleted = db.Column(db.BOOLEAN)
