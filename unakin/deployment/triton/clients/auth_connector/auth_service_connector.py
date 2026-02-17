from typing import Annotated, Optional

import requests
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

# TODO can't find a way to inject this from top level too...
OAUTH2_SCHEME = OAuth2PasswordBearer(tokenUrl="/auth/token")

class TokenResponse(BaseModel):
    access_token: str
    token_type: str

class UserResponse(BaseModel):
    id: int
    email: str
    first_name: str
    last_name: str
    permission: list[str]

class AuthServiceConnector(APIRouter):
  def __init__(self, *args,
                auth_server_url: str,
                frontend_url: str,
                  **kwargs):
    super().__init__(*args, **kwargs)

    self.auth_server_url = auth_server_url
    self.frontend_url = frontend_url

    self._add_own_routes()

  def _add_own_routes(self):
     self.add_api_route(
        path="/token",
        endpoint=self.r_login_for_access_token,
        methods=["POST"],
        response_model=TokenResponse,
      )
     self.add_api_route(
        path="/me",
        endpoint=self.r_get_current_user,
        methods=["GET"],
        response_model=UserResponse,
      )
     
  # routes

  async def r_login_for_access_token(
    self,
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
  ) -> TokenResponse:
    req = requests.post(
        url=self.auth_server_url + "/token",
        data={
            "username": form_data.username,
            "password": form_data.password
        }
    )

    # if the request is not successful, raise an exception
    if req.status_code != 200:
        raise HTTPException(
            status_code=req.status_code
        )

    return req.json()

  async def r_get_current_user(self, token: Annotated[str, Depends(OAUTH2_SCHEME)]) -> UserResponse:
    req = requests.get(
        url=self.auth_server_url + "/me",
        headers={
            "Authorization": f"Bearer {token}"
        }
    )

    # if the request is not successful, raise an exception
    if req.status_code != 200:
        raise HTTPException(
            status_code=req.status_code
        )

    return req.json()
