import json

import requests
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

from Utils import PrintJson

_ = load_dotenv(find_dotenv())

client = OpenAI()

"""
接口
@RestController
public class LLMController {
    private static final Logger log = LoggerFactory.getLogger(LLMController.class);

    @GetMapping("/v1/llm/place")
    public String place(String location, String city) {
        log.info("=== Function Calling place ===");
        if (city.equals("上海") && location.equals("黄浦区")) {
            return "洛克公园";
        }
        return "无篮球场";
    }

    @GetMapping("/v1/llm/distance")
    public String distance(String location, String city) {
        log.info("=== Function Calling distance ===");
        if (city.equals("上海") && location.equals("洛克公园")) {
            return "100米";
        }
        return "无法找到";
    }
}
"""


def func_call(messages, model="gpt-4o"):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        seed=1024,
        tool_choice="auto",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_place",
                    "description": "返回地点",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "区域",
                            },
                            "city": {
                                "type": "string",
                                "description": "城市",
                            }
                        },
                        "required": ["location", "city"],
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_distance",
                    "description": "返回距离",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "区域",
                            },
                            "city": {
                                "type": "string",
                                "description": "城市",
                            }
                        },
                        "required": ["location", "city"],
                    }
                }
            }
        ]
    )
    print(response)
    return response.choices[0].message


def get_place(location, city):
    url = f"http://localhost:8080/v1/llm/place?location={location}&city={city}"
    print(url)
    r = requests.get(url)
    result = r.content.decode()
    return result


def get_distance(location, city):
    url = f"http://localhost:8080/v1/llm/distance?location={location}&city={city}"
    print(url)
    r = requests.get(url)
    result = r.content.decode()
    return result


def main():
    # prompt = "1+1=?"
    prompt = "在上海黄浦区附近打篮球，推荐几个篮球场并告诉我距离多远"
    messages = [
        {"role": "system", "content": "你是一个地图工具"},
        {"role": "user", "content": prompt}
    ]
    response = func_call(messages)
    messages.append(response)

    print("===== chatGPT 回复 =====")

    PrintJson.print_json(response)
    while response.tool_calls is not None:
        for tool_call in response.tool_calls:
            args = json.loads(tool_call.function.arguments)

            print("函数参数展开: ")
            PrintJson.print_json(args)

            if tool_call.function.name == "get_place":
                print("Func Call: get_place")
                result = get_place(**args)
            elif tool_call.function.name == "get_distance":
                print("Func Call: get_distance")
                result = get_distance(**args)

            print("===== 函数返回 =====")
            PrintJson.print_json(result)

            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_call.function.name,
                "content": str(result)
            })
        response = func_call(messages)
        messages.append(response)

    print("===== 最终回复 =====")
    print(response.content)
    print("===== 对话历史 =====")
    PrintJson.print_json(messages)


if __name__ == '__main__':
    main()
