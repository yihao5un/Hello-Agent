from typing import Optional, Dict, Any

from pydantic import BaseModel, Field

"""
定义Action的输出格式
会被填在 prompt 的最后 {format_instructions}
Langchain 中的 pydantic parser 直接处理的格式 最后以Json格式输出
"""
class Action(BaseModel):
    name: str = Field(description="Tool name")
    args: Optional[Dict[str, Any]] = Field(description="Tool input arguments, containing arguments names and values")
