from typing import Optional, Dict, Any

from pydantic import BaseModel, Field

"""
定义Action的输出格式
会被填在 prompt 的最后 {format_instructions}
Langchain 中的 pydantic parser 直接处理的格式 最后以Json格式输出
"""

"""
这段代码定义了一个名为`Action`的类，它继承自`BaseModel`。这通常意味着您正在使用Pydantic库来定义数据模型，Pydantic是一个用于数据验证和设置的库，常用于FastAPI等现代Python web框架中。
### `Action`类结构解析：
1. **`name: str = Field(description="Tool name")`**:
   - `name` 是一个类属性，它的类型被指定为 `str`（字符串）。
   - `Field` 是Pydantic提供的一个函数，用于为模型字段添加额外的元数据或者修改默认行为。在这里，它用来提供一个描述性的文档字符串，说明这个字段代表的是“Tool name”（工具名称）。如果不特别指定，默认情况下字段是必填的。
2. **`args: Optional[Dict[str, Any]] = Field(description="Tool input arguments, containing arguments names and values")`**:
   - `args` 是另一个类属性，它的类型被声明为 `Optional[Dict[str, Any]]`。这意味着`args`可以是字典类型，键为字符串，值为任意类型（`Any`），也可以是`None`。`Optional`是一个类型提示，表示这个字段可以是其后面类型的值，也可以是`None`。
   - 同样使用了`Field`函数来提供描述，说明这个字段是用来存储“Tool input arguments”（工具输入参数）的，这些参数包含参数名和对应的值。
总结来说，`Action`类定义了一个数据模型，用于表示某个动作或工具的执行信息。它包含两个主要部分：`name`表示工具的名称，是必须提供的字符串；而`args`是一个可选参数，用于传递给该工具的输入参数，这些参数以字典形式存在，键为参数名，值为参数值，可以是任意类型。这样的设计使得在处理如API请求参数或配置信息时，能够方便地进行数据验证和序列化/反序列化操作。
"""


class Action(BaseModel):
    name: str = Field(description="Tool name")
    args: Optional[Dict[str, Any]] = Field(description="Tool input arguments, containing arguments names and values")
