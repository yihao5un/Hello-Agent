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


class AutoGPT:
    """AutoGPT：基于Langchain实现"""

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
