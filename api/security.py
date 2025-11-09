# api/security.py
# 간단한 SHA256 해시 유틸 (배포에서는 bcrypt/argon2 등으로 교체 권장)
import hashlib

def hash_password(plain: str) -> str:
    if plain is None:
        return ""
    return hashlib.sha256(plain.encode('utf-8')).hexdigest()
