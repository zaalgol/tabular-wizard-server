# from functools import wraps
# from app import db

# def with_session(f):
#     @wraps(f)
#     def decorated_function(*args, **kwargs):
#         # Check if a session is already provided
#         if 'session' not in kwargs or kwargs['session'] is None:
#             kwargs['session'] = db.session
#         return f(*args, **kwargs)
#     return decorated_function