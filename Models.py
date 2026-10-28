import re
from typing import Optional, Literal, Dict, Any

from pydantic import BaseModel, Field, field_validator


TABLE_CODE = 'llm'


class RequestModel(BaseModel):
    request_id: str = Field(..., alias="RequestId")
    system_prompt: Optional[str] = Field(None, alias="SystemPrompt")
    user_prompt: str = Field(..., alias="UserPrompt")

    model: str = Field('gpt-5.4-mini', alias="Model")
    reasoning_effort: Optional[Literal['none', 'minimal', 'low', 'medium', 'high', 'xhigh']] = \
        Field(None, alias="ReasoningEffort", description="None -> DEFAULT_REASONING_EFFORT")
    temperature: Optional[float] = Field(None, ge=0., le=2., alias="Temperature",
                                         description="None -> DEFAULT_TEMPERATURE")

    @field_validator('request_id', mode='before')
    def _validate_id(cls, val) -> str:
        if val is None or not isinstance(val, str):
            raise ValueError(f"Id must be string!")

        ptrn_text = rf"^{TABLE_CODE}-[0-9a-f]{{8}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{12}}$"
        pattern = re.compile(ptrn_text, re.IGNORECASE)

        if not pattern.fullmatch(val):
            raise ValueError(f"Invalid id ({val})!")

        return val


class ResponseModel(BaseModel):
    request_id: str = Field(..., alias="RequestId")
    status: Literal['Success', 'Error'] = Field('Success', alias='Status')
    error_message: Optional[str] = Field(None, alias="ErrorMessage")

    response: Dict[str, Any] = Field(..., alias="Response")
