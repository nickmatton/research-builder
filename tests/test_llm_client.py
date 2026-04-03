"""Tests for the LLM client wrapper."""

import pytest

from research_builder.llm.client import LLMClient, ToolExit
from research_builder.config import Config


class TestToolExit:
    def test_carries_result(self):
        result = {"status": "success", "summary": "done"}
        exc = ToolExit(result=result)
        assert exc.result == result
        assert str(exc) == "Tool loop exit"

    def test_is_exception(self):
        with pytest.raises(ToolExit):
            raise ToolExit(result={"status": "failure"})


class TestLLMClientInit:
    def test_creates_with_config(self):
        config = Config(model="claude-opus-4-6")
        client = LLMClient(config)
        assert client.model == "claude-opus-4-6"
        assert client.client is not None
