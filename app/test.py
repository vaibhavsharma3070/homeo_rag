import jwt
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

APPLE_PUBLIC_KEYS_URL = "https://appleid.apple.com/auth/keys"
APPLE_AUDIENCE = "com.homeopathy.app"  # your bundle id


class AppleLoginRequest(BaseModel):
    identity_token: str


def get_apple_public_key(kid: str):
    response = requests.get(APPLE_PUBLIC_KEYS_URL)
    keys = response.json()["keys"]

    for key in keys:
        if key["kid"] == kid:
            return jwt.algorithms.RSAAlgorithm.from_jwk(key)

    return None


@app.post("/apple-login")
def apple_login(data: AppleLoginRequest):
    identity_token = data.identity_token

    try:
        # Step 1: Get token header
        header = jwt.get_unverified_header(identity_token)
        kid = header.get("kid")

        # Step 2: Get Apple's public key
        public_key = get_apple_public_key(kid)
        if not public_key:
            raise HTTPException(status_code=400, detail="Public key not found")

        # Step 3: Verify token
        payload = jwt.decode(
            identity_token,
            public_key,
            algorithms=["RS256"],
            audience=APPLE_AUDIENCE,
            issuer="https://appleid.apple.com",
        )

        apple_user_id = payload.get("sub")
        email = payload.get("email")

        # TODO: Check if user exists in DB
        # If not, create user
        # If exists, login user

        return {
            "status": "success",
            "apple_user_id": apple_user_id,
            "email": email,
        }

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=400, detail="Token expired")

    except jwt.InvalidTokenError:
        raise HTTPException(status_code=400, detail="Invalid token")

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
