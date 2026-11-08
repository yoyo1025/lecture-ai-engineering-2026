"""エージェントの配線テスト。

TestModel は実際のLLMを呼ばない偽モデルで、登録されたツールを全部呼んでから
ダミーの回答を返す。APIキーも利用枠も使わずに、
「ツールが登録され、呼ばれ、結果がモデルに戻るか」を確かめられる。
"""

import importlib
import re

from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from app.agent import agent
from app import tools

agent_module = importlib.import_module("app.agent")

BASE_TOOLS = {"search_glossary", "get_lecture_summary", "make_quiz"}


def _parts(result, kind):
    return [p for m in result.all_messages() for p in m.parts if isinstance(p, kind)]


def test_base_tools_are_registered_and_called():
    model = TestModel()
    with agent.override(model=model):
        result = agent.run_sync("LoRAとは？")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert BASE_TOOLS <= offered  # 演習でツールを足しても壊れないよう「含む」で判定

    called = {p.tool_name for p in _parts(result, ToolCallPart)}
    assert BASE_TOOLS <= called
    # 承認が必要なツール（requires_approval=True）は、実行されずに承認待ちになるので戻り値が無い
    waiting = len(result.output.approvals) if isinstance(result.output, DeferredToolRequests) else 0
    assert len(_parts(result, ToolReturnPart)) + waiting == len(_parts(result, ToolCallPart))
    assert result.output


def test_glossary_result_reaches_the_model():
    model = TestModel(call_tools=["search_glossary"])
    with agent.override(model=model):
        result = agent.run_sync("RAGとは？")
    returns = _parts(result, ToolReturnPart)
    assert returns and returns[0].tool_name == "search_glossary"


def test_cli_approval_saves_learning_log(tmp_path, monkeypatch, capsys):
    prompts = []
    monkeypatch.setattr(tools, "LOG_DIR", tmp_path / "learning_log")
    monkeypatch.setattr(agent_module, "build_model", iter([
        TestModel(call_tools=["save_learning_log"]),
        TestModel(call_tools=[]),
    ]).__next__)
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "y")
    monkeypatch.setattr("sys.argv", ["app.agent", "学んだことを記録して"])

    agent_module.main()

    log_files = list((tmp_path / "learning_log").glob("learning_log_*.txt"))
    assert len(log_files) == 1
    log_path = log_files[0]
    assert re.fullmatch(r"learning_log_\d{8}_\d{6}_\d{6}\.txt", log_path.name)
    assert log_path.read_text(encoding="utf-8") == "a\n"
    assert prompts == ["実行を許可しますか？ [y/N]: "]
    assert "実行が承認待ちです" in capsys.readouterr().out


def test_cli_denial_does_not_save_learning_log(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "LOG_DIR", tmp_path / "learning_log")
    monkeypatch.setattr(agent_module, "build_model", iter([
        TestModel(call_tools=["save_learning_log"]),
        TestModel(call_tools=[]),
    ]).__next__)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    monkeypatch.setattr("sys.argv", ["app.agent", "学んだことを記録して"])

    agent_module.main()

    assert not (tmp_path / "learning_log").exists()
