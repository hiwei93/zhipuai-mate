from typing import List, Optional, Any, Dict, Mapping, Union

import zhipuai
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.pydantic_v1 import Field, root_validator, BaseModel


def convert_message_to_dict(message: BaseMessage) -> dict:
    message_dict: Dict[str, Any]
    if isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    else:
        raise TypeError(f"Got unknown type {message}")
    return message_dict


def convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    role = _dict.get("role")
    if role == 'assistant':
        return AIMessage(content=_dict.get("content", ""), additional_kwargs={})
    raise TypeError(f"Got unknown type {_dict}")


class ChatZhipuAI(BaseChatModel):
    client: Any = None
    zhipuai_api_key: Optional[str] = None
    model_name: str = "chatglm_turbo"
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""

    temperature: float = 0.95
    top_p: float = 0.7

    @root_validator(pre=True)
    def build_client(cls, values: dict) -> dict:
        if not values.get('client'):
            values['client'] = zhipuai.ZhipuAI(
                api_key=values.get('zhipuai_api_key')
            )
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        return params

    def _create_chat_result(self, response: Union[dict, BaseModel]) -> ChatResult:
        generations = []
        if not isinstance(response, dict):
            response = response.dict()
        for choice in response["choices"]:
            message = convert_dict_to_message(choice["message"])
            generation_info = dict(finish_reason=choice.get("finish_reason"))
            gen = ChatGeneration(
                message=message,
                generaion_info=generation_info,
            )
            generations.append(gen)
        token_usage = response.get("usage", {})
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model_name,
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> ChatResult:
        message_dicts = [convert_message_to_dict(m) for m in messages]
        params = {
            **self.model_kwargs,
            **self._default_params,
            **kwargs
        }
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message_dicts,
            temperature=params["temperature"],
            top_p=params["top_p"]
        )
        response = response.dict()
        return self._create_chat_result(response)

    @property
    def _llm_type(self) -> str:
        return "zhipuai-chat"
