from typing import List, Tuple

from langchain.memory import ConversationTokenBufferMemory
from langchain.memory.chat_memory import BaseChatMemory
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.schema.output_parser import StrOutputParser
from langchain.tools.base import BaseTool
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.tools import render_text_description
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from Agent.Action import Action
from Utils.CallbackHandlers import *

"""AutoGPT: 基于Langchain实现"""

"""
这段代码定义了一个名为`AutoGPT`的类，它利用LangChain框架以及相关的库（如Pydantic、Langchain的Tools、内存管理、输出解析器等）来构建一个能够自动执行任务、决策并利用一系列工具的AI代理。
以下是关键部分的解析：

### 类初始化 (`__init__`方法)
- **大模型 (`llm`)**: 接收一个基于聊天的模型实例，如`ChatOpenAI`，作为决策和生成逻辑的基础。
- **工具集 (`tools`)**: 一个包含多个`BaseTool`实例的列表，代表AI可以调用的不同功能或服务。
- **工作目录 (`work_dir`)**: 指定AI进行操作时参考的根目录路径。
- **主提示文件 (`main_prompt_file`)**: 包含AI如何自我引导和决策的主要逻辑的文本文件。
- **最大思考步数 (`max_thought_steps`)**: 限制AI解决问题时的递归深度，防止无限循环。

### 核心功能
1. **输出解析**: 使用`PydanticOutputParser`和`OutputFixingParser`确保AI输出能被解析为具体的`Action`对象，便于执行。
2. **初始化提示模板**: 通过读取`main_prompt_file`内容，并结合历史对话和工具描述，构建一个复合的提示模板，用于引导AI的思考和决策。
3. **链式逻辑构造**: 利用LangChain的链式API (`|`操作符) 构建从提示生成到模型响应再到解析输出的整个处理流程。
4. **思考与执行循环 (`__step`方法)**: 在给定的最大思考步数内，AI会根据当前任务、已有的对话历史和短时记忆（由`ConversationTokenBufferMemory`管理）来决定下一步行动。每一步都包括生成行动指令、解析指令、执行工具操作并记录结果。
5. **工具执行 (`__exec_action`方法)**: 根据解析出的`Action`找到相应的工具并执行，捕获可能发生的异常（如参数验证失败或执行错误）。
6. **运行 (`run`方法)**: 接受用户任务、对话历史和是否开启详细日志模式，启动整个决策和执行流程。过程中，AI会不断迭代思考，更新短时记忆，并最终返回执行结果或达到思考步数上限时的错误信息。

### 技术栈亮点
- **LangChain**: 提供了一套高级抽象，简化了使用语言模型和构建复杂应用的过程。
- **Pydantic**: 用于数据验证和定义数据模型，确保AI输出的结构化和一致性。
- **CallbackHandlers**: 实现了日志和调试功能，增加了运行时的透明度和可控性。
整体上，这个`AutoGPT`类展示了一个高度模块化和可扩展的设计，旨在实现自主决策和高效任务执行的AI代理。
"""


class AutoGPT:
    """
    在Python中，`@staticmethod`是一个装饰器，用于标记一个方法为静态方法。静态方法属于类本身而不是类的实例，这意味着你不需要创建类的实例就可以直接通过类名来调用静态方法。
    静态方法不接收隐式的第一个参数（比如实例方法中的`self`），也不与类的实例绑定，因此在静态方法内部不能直接访问实例变量或调用实例方法。
    静态方法主要用于与类相关的操作，而这些操作并不需要特定的实例状态。它可以被视作与类有关的全局函数，但将其定义在类中是为了逻辑上更好的组织和命名空间的封装。
    例如：
    ```python
    class MyClass:
        @staticmethod
        def my_static_method(arg1, arg2):
            # 这里可以直接使用arg1, arg2，但不能使用self
            return arg1 + arg2
    
    # 调用静态方法，无需创建类的实例
    result = MyClass.my_static_method(1, 2)
    print(result)  
    # 输出：3
    ```
    在这个例子中，`my_static_method`是一个静态方法，可以直接通过类名`MyClass.my_static_method()`来调用，而不需要创建`MyClass`的实例。
    ===================================================================================================================
    在Java中，与Python的`@staticmethod`相对应的概念是静态方法（static method）。在Java中，你使用`static`关键字来声明一个方法为静态方法。和Python一样，Java的静态方法也属于类本身，而不是类的某个特定实例，因此可以通过类名直接调用，而无需创建类的实例。
    例如，Java中的静态方法看起来像这样：
    ```java
    public class MyClass {
        public static int myStaticMethod(int arg1, int arg2) {
            // 这里可以直接使用arg1和arg2，但不能访问非静态成员变量或调用非静态方法
            return arg1 + arg2;
        }
    }
    // 调用静态方法，无需创建类的实例
    int result = MyClass.myStaticMethod(1, 2);
    System.out.println(result);  // 输出：3
    ```
    在这两个语言中，静态方法的主要用途和行为都是相似的，都是用于提供那些不依赖于类实例状态的操作。
    """
    """
    这段代码定义了一个名为`__format_short_term_memory`的静态方法，它属于某个类（虽然上下文未给出类名，但从使用`@staticmethod`装饰器推测）。此方法的功能是将短期对话记忆（`memory`）中的消息内容转换为一个字符串，其中每一项消息内容之间用换行符分隔。
    - **参数**:
      - `memory: BaseChatMemory`: 表示输入参数`memory`是一个`BaseChatMemory`类型的对象，这是LangChain中用于存储聊天记录的基类。这个对象中包含了聊天的交互历史，每个交互通常由一个`message`表示。
    - **方法细节**:
      - `messages = memory.chat_memory.messages`: 获取`memory`对象中的所有消息记录。这里假设`BaseChatMemory`有一个属性`chat_memory`，它又有一个属性`messages`，存储了所有的消息。
    - **列表推导式**:
      - `[messages[i].content for i in range(1, len(messages))]`: 从消息列表中（除了第一条消息外，因为索引从1开始），提取每个消息的内容（`.content`属性）到一个新的列表`string_messages`中。这样做通常是因为第一条消息可能是系统消息或无关紧要的上下文，但具体原因需依据实际应用场景而定。
    - **返回值**:
      - `return "\n".join(string_messages)`: 将`string_messages`列表中的所有字符串元素用换行符`\n`连接起来，形成一个单一的字符串，表示所有聊天消息的连续文本内容，每条消息占一行。
    **注意**：由于这是一个静态方法，它不依赖于类的实例状态，这意味着在调用该方法时，不需要类的实例，可以直接通过类名调用，如`MyClass.__format_short_term_memory(some_memory_instance)`。不过，由于该方法名前有两个下划线`__`，这通常表明它是类的私有方法，意在限制外部直接访问，通常只在类内部使用。
    """
    # 没用原生的 而是自己拼的
    @staticmethod
    def __format_short_term_memory(memory: BaseChatMemory) -> str:
        messages = memory.chat_memory.messages
        string_messages = [messages[i].content for i in range(1, len(messages))]
        return "\n".join(string_messages)

    """
    初始化
    """
    def __init__(
            self,
            # 大模型
            llm: BaseChatModel,
            # 一组工具列表
            tools: List[BaseTool],
            # 文件工作目录 prompt中保证完整性
            work_dir: str,
            # 主prompt文件 main.txt
            main_prompt_file: str,
            # 约定最多思考多少步 默认10步
            max_thought_steps: Optional[int] = 10,
    ):
        self.llm = llm
        self.tools = tools
        self.work_dir = work_dir
        self.main_prompt_file = main_prompt_file
        self.max_thought_steps = max_thought_steps

        # OutputFixingParser： 如果输出格式不正确，尝试修复
        self.output_parser = PydanticOutputParser(pydantic_object=Action)
        self.robust_parser = OutputFixingParser.from_llm(
            parser=self.output_parser,
            llm=ChatOpenAI(
                model="gpt-4o",
                temperature=0,
                model_kwargs={"seed": 42}
            )
        )

        self.__init_prompt_templates()
        self.__init_chains()
        # 创建回调函数 为了显示中间步骤
        self.verbose_handler = ColoredPrintHandler(color=THOUGHT_COLOR)

    """
    加载 prompt 文件
    """
    def __init_prompt_templates(self):
        with open(self.main_prompt_file, 'r', encoding='utf-8') as f:
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    # 存放历史多轮对话
                    MessagesPlaceholder(variable_name="chat_history"),
                    # 当前轮 main_prompt_file
                    # HumanMessagePromptTemplate 不支持文件加载 但可以手动打开直接读文件成字符串
                    HumanMessagePromptTemplate.from_template(f.read()),
                ]
            ).partial(
                # partial 启动之前的都写上了
                work_dir=self.work_dir,
                tools=render_text_description(self.tools),
                # tool_names langchain原生模版中有
                tool_names=','.join([tool.name for tool in self.tools]),
                # 选择执行的动作/工具 输出的格式
                format_instructions=self.output_parser.get_format_instructions(),
            )

    def __init_chains(self):
        # 主流程的chain
        self.main_chain = (self.prompt | self.llm | StrOutputParser())

    def __find_tool(self, tool_name: str) -> Optional[BaseTool]:
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None

    def __step(self,
               task,
               short_term_memory,
               chat_history,
               verbose=False
               ) -> Tuple[Action, str]:

        """执行一步思考"""
        response = ""
        for s in self.main_chain.stream({
            "input": task,
            """ 
            agent_scratchpad 记录短时记忆, 随着思考会越来越多
            每一步走的思考过程,选择工具的动作和执行结果会放在这里
            例: 
            思考第二步的时候 第一步的会放在 agent_scratchpad 
            思考第三步的时候 第一步和第二步的结果会放在 agent_scratchpad
            """
            "agent_scratchpad": self.__format_short_term_memory(
                short_term_memory
            ),
            "chat_history": chat_history.messages,
        }, config={
            "callbacks": [
                self.verbose_handler
            ] if verbose else []
        }):
            response += s

        action = self.robust_parser.parse(response)
        return action, response

    def __exec_action(self, action: Action) -> str:
        # 查找工具
        tool = self.__find_tool(action.name)
        if tool is None:
            observation = (
                f"Error: 找不到工具或指令 '{action.name}'. "
                f"请从提供的工具/指令列表中选择，请确保按对顶格式输出。"
            )
        else:
            try:
                # 执行工具
                observation = tool.run(action.args)
            except ValidationError as e:
                # 工具的入参异常
                observation = (
                    f"Validation Error in args: {str(e)}, args: {action.args}"
                )
            except Exception as e:
                # 工具执行异常
                observation = f"Error: {str(e)}, {type(e).__name__}, args: {action.args}"

        return observation

    """
    运行
    """
    def run(
            self,
            task: str,
            chat_history: ChatMessageHistory,
            verbose=False
    ) -> str:
        """
        运行智能体
        :param task: 用户任务
        :param chat_history: 对话上下文（长时记忆）
        :param verbose: 是否显示详细信息
        """
        # 初始化短时记忆: 记录推理过程
        short_term_memory = ConversationTokenBufferMemory(
            llm=self.llm,
            max_token_limit=4000,
        )

        # 思考步数
        thought_step_count = 0

        reply = ""

        # 开始逐步思考
        while thought_step_count < self.max_thought_steps:
            if verbose:
                self.verbose_handler.on_thought_start(thought_step_count)

            # 执行一步思考 调用 prompt chain
            action, response = self.__step(
                task=task,
                short_term_memory=short_term_memory,
                chat_history=chat_history,
                verbose=verbose,
            )

            # 如果是结束指令，执行最后一步
            if action.name == "FINISH":
                reply = self.__exec_action(action)
                break

            # 执行动作
            observation = self.__exec_action(action)

            if verbose:
                self.verbose_handler.on_tool_end(observation)

            # 更新短时记忆
            short_term_memory.save_context(
                {"input": response},
                {"output": "\n返回结果:\n" + observation}
            )

            thought_step_count += 1

        if thought_step_count >= self.max_thought_steps:
            # 如果思考步数达到上限，返回错误信息
            reply = "抱歉，我没能完成您的任务。"

        # 更新长时记忆
        chat_history.add_user_message(task)
        chat_history.add_ai_message(reply)
        return reply
