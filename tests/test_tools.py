"""Tests for sub-agent tool executor."""

import asyncio
from pathlib import Path

import pytest

from research_builder.llm.client import ToolExit
from research_builder.sub_agent.tools import ALL_TOOLS, ToolExecutor

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "test_paper.pdf"


@pytest.fixture
def work_dir(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    return tmp_path


@pytest.fixture
def executor(work_dir):
    return ToolExecutor(
        work_dir=work_dir,
        paper_path=FIXTURE_PDF,
        bash_timeout=10,
    )


# --- Tool definitions ---


class TestToolDefinitions:
    def test_all_tools_have_names(self):
        names = {t["name"] for t in ALL_TOOLS}
        assert names == {
            "read_file", "write_file", "edit_file",
            "bash", "read_paper_section", "report_result",
        }

    def test_all_tools_have_schemas(self):
        for tool in ALL_TOOLS:
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"


# --- read_file ---


class TestReadFile:
    @pytest.mark.asyncio
    async def test_read_existing(self, executor, work_dir):
        (work_dir / "test.txt").write_text("hello world")
        result = await executor.execute("read_file", {"path": "test.txt"})
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_read_missing(self, executor):
        result = await executor.execute("read_file", {"path": "nope.txt"})
        assert "Error" in result
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_read_absolute(self, executor, work_dir):
        f = work_dir / "abs.txt"
        f.write_text("absolute")
        result = await executor.execute("read_file", {"path": str(f)})
        assert result == "absolute"


# --- write_file ---


class TestWriteFile:
    @pytest.mark.asyncio
    async def test_write_new(self, executor, work_dir):
        result = await executor.execute("write_file", {"path": "out.txt", "content": "data"})
        assert "Wrote" in result
        assert (work_dir / "out.txt").read_text() == "data"

    @pytest.mark.asyncio
    async def test_write_creates_dirs(self, executor, work_dir):
        result = await executor.execute("write_file", {"path": "sub/dir/f.txt", "content": "nested"})
        assert "Wrote" in result
        assert (work_dir / "sub" / "dir" / "f.txt").read_text() == "nested"

    @pytest.mark.asyncio
    async def test_write_overwrites(self, executor, work_dir):
        (work_dir / "exist.txt").write_text("old")
        await executor.execute("write_file", {"path": "exist.txt", "content": "new"})
        assert (work_dir / "exist.txt").read_text() == "new"


# --- edit_file ---


class TestEditFile:
    @pytest.mark.asyncio
    async def test_edit_replace(self, executor, work_dir):
        (work_dir / "code.py").write_text("x = 1\ny = 2\nz = 3\n")
        result = await executor.execute("edit_file", {
            "path": "code.py",
            "old_text": "y = 2",
            "new_text": "y = 42",
        })
        assert "Edited" in result
        assert (work_dir / "code.py").read_text() == "x = 1\ny = 42\nz = 3\n"

    @pytest.mark.asyncio
    async def test_edit_not_found(self, executor, work_dir):
        (work_dir / "code.py").write_text("x = 1")
        result = await executor.execute("edit_file", {
            "path": "code.py",
            "old_text": "NOPE",
            "new_text": "whatever",
        })
        assert "Error" in result
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_edit_ambiguous(self, executor, work_dir):
        (work_dir / "code.py").write_text("x = 1\nx = 1\n")
        result = await executor.execute("edit_file", {
            "path": "code.py",
            "old_text": "x = 1",
            "new_text": "x = 2",
        })
        assert "Error" in result
        assert "2 times" in result

    @pytest.mark.asyncio
    async def test_edit_missing_file(self, executor):
        result = await executor.execute("edit_file", {
            "path": "nope.py",
            "old_text": "a",
            "new_text": "b",
        })
        assert "Error" in result
        assert "not found" in result


# --- bash ---


class TestBash:
    @pytest.mark.asyncio
    async def test_echo(self, executor):
        result = await executor.execute("bash", {"command": "echo hello"})
        assert "hello" in result
        assert "exit code: 0" in result

    @pytest.mark.asyncio
    async def test_cwd_is_work_dir(self, executor, work_dir):
        result = await executor.execute("bash", {"command": "pwd"})
        assert str(work_dir) in result

    @pytest.mark.asyncio
    async def test_stderr(self, executor):
        result = await executor.execute("bash", {"command": "echo err >&2"})
        assert "err" in result

    @pytest.mark.asyncio
    async def test_nonzero_exit(self, executor):
        result = await executor.execute("bash", {"command": "exit 1"})
        assert "exit code: 1" in result

    @pytest.mark.asyncio
    async def test_timeout(self, executor):
        result = await executor.execute("bash", {"command": "sleep 30", "timeout": 1})
        assert "timed out" in result


# --- read_paper_section ---


class TestReadPaperSection:
    @pytest.mark.asyncio
    async def test_read_single_page(self, executor):
        result = await executor.execute("read_paper_section", {"start_page": 1})
        assert "Test Paper" in result
        assert "pages 1-1 of 3" in result

    @pytest.mark.asyncio
    async def test_read_range(self, executor):
        result = await executor.execute("read_paper_section", {"start_page": 2, "end_page": 3})
        assert "Methods" in result
        assert "Results" in result

    @pytest.mark.asyncio
    async def test_out_of_bounds(self, executor):
        result = await executor.execute("read_paper_section", {"start_page": 99})
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_missing_paper(self, work_dir):
        ex = ToolExecutor(work_dir=work_dir, paper_path=Path("/nonexistent.pdf"))
        result = await ex.execute("read_paper_section", {"start_page": 1})
        assert "Error" in result


# --- report_result ---


class TestReportResult:
    @pytest.mark.asyncio
    async def test_raises_tool_exit(self, executor):
        with pytest.raises(ToolExit) as exc_info:
            await executor.execute("report_result", {
                "status": "success",
                "summary": "Everything works",
                "outputs": [{"name": "model", "file_path": "outputs/model.pt"}],
                "test_report": {"tests_run": 2, "tests_passed": 2, "tests_failed": 0},
                "attempts_used": 1,
            })
        result = exc_info.value.result
        assert result["status"] == "success"
        assert result["summary"] == "Everything works"
        assert len(result["outputs"]) == 1

    @pytest.mark.asyncio
    async def test_failure_with_spec_issue(self, executor):
        with pytest.raises(ToolExit) as exc_info:
            await executor.execute("report_result", {
                "status": "failure",
                "summary": "Dataset URL is broken",
                "is_spec_issue": True,
                "diagnostics": {"url": "https://example.com/data.tar.gz"},
            })
        result = exc_info.value.result
        assert result["status"] == "failure"
        assert result["is_spec_issue"] is True


# --- unknown tool ---


class TestUnknownTool:
    @pytest.mark.asyncio
    async def test_unknown(self, executor):
        result = await executor.execute("made_up_tool", {"foo": "bar"})
        assert "Error" in result
        assert "unknown tool" in result
