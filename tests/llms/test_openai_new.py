import unittest

from types import SimpleNamespace
from unittest.mock import MagicMock

from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_reasoning_item import ResponseReasoningItem, Summary

from memos.configs.llm import OpenAIResponsesLLMConfig
from memos.llms.openai_new import OpenAIResponsesLLM


class TestOpenAIResponsesLLM(unittest.TestCase):
    def setUp(self):
        config = OpenAIResponsesLLMConfig(
            model_name_or_path="gpt-4.1-nano",
            api_key="test-key",
        )
        self.llm = OpenAIResponsesLLM(config)

    @staticmethod
    def output_message(text: str) -> ResponseOutputMessage:
        return ResponseOutputMessage(
            id="msg_1",
            content=[
                ResponseOutputText(
                    annotations=[],
                    text=text,
                    type="output_text",
                )
            ],
            role="assistant",
            status="completed",
            type="message",
        )

    def test_generate_returns_text_without_reasoning_output(self):
        response = SimpleNamespace(
            output=[self.output_message("plain answer")],
            output_text="plain answer",
        )
        self.llm.client.responses.create = MagicMock(return_value=response)

        result = self.llm.generate([{"role": "user", "content": "hello"}])

        self.assertEqual(result, "plain answer")

    def test_generate_includes_reasoning_summary_when_present(self):
        reasoning = ResponseReasoningItem(
            id="reasoning_1",
            summary=[Summary(text="brief thought", type="summary_text")],
            type="reasoning",
            status="completed",
        )
        response = SimpleNamespace(
            output=[reasoning, self.output_message("final answer")],
            output_text="final answer",
        )
        self.llm.client.responses.create = MagicMock(return_value=response)

        result = self.llm.generate([{"role": "user", "content": "hello"}])

        self.assertEqual(result, "<think>brief thought</think>final answer")


if __name__ == "__main__":
    unittest.main()
