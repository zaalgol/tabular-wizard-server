import bcrypt

class PasswordHasher:
    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash a password for storing.
        """
        # Convert the password to bytes and hash it
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        return hashed.decode('utf-8')
    
    @staticmethod
    def check_password(hashed_password: str, user_password: str) -> bool:
        """
        Check a hashed password. Return True if the password matches, False otherwise.
        """
        # Convert the hashed password and user password to bytes, then check them
        return bcrypt.checkpw(user_password.encode('utf-8'), hashed_password.encode('utf-8'))
