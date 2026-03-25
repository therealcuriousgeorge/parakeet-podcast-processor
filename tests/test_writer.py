"""Tests for BlogWriter — provider dispatch, slug generation, grading, social posts."""

import re
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from p3.database import P3Database
from p3.writer import BlogWriter


@pytest.fixture
def db(tmp_path):
    database = P3Database(str(tmp_path / "test.duckdb"))
    yield database
    database.close()


SAMPLE_DIGEST = {
    "episode_title": "AI in 2026",
    "podcast_title": "Tech Talk",
    "full_summary": "A discussion about AI trends.",
    "key_topics": ["LLMs", "agents"],
    "themes": ["automation"],
    "quotes": ["The future is here."],
    "startups": ["OpenAI"],
}

FAKE_POST = "AI is transforming how we work. The implications are profound and far-reaching."
FAKE_GRADE_RESPONSE = "GRADE: A-\nSCORE: 92\nFEEDBACK: Strong hook, clear argument, good transitions."


# ── Slug generation ───────────────────────────────────────────────────────────

class TestSlugGeneration:
    def test_basic_slug(self, db):
        w = BlogWriter(db)
        assert w._generate_slug("Hello World") == "hello-world"

    def test_removes_special_chars(self, db):
        w = BlogWriter(db)
        slug = w._generate_slug("AI's Impact on Software!")
        assert re.match(r'^[a-z0-9-]+$', slug)

    def test_collapses_multiple_hyphens(self, db):
        w = BlogWriter(db)
        slug = w._generate_slug("AI  --  LLMs")
        assert "--" not in slug

    def test_no_leading_or_trailing_hyphens(self, db):
        w = BlogWriter(db)
        slug = w._generate_slug("  AI trends  ")
        assert not slug.startswith("-")
        assert not slug.endswith("-")


# ── Grade response parsing ────────────────────────────────────────────────────

class TestGradeResponseParsing:
    def test_parses_letter_grade(self, db):
        w = BlogWriter(db)
        with patch.object(w, '_generate_with_llm', return_value=FAKE_GRADE_RESPONSE):
            result = w._grade_blog_post("some blog post content")
        assert result["grade"] == "A-"

    def test_parses_numerical_score(self, db):
        w = BlogWriter(db)
        with patch.object(w, '_generate_with_llm', return_value=FAKE_GRADE_RESPONSE):
            result = w._grade_blog_post("some blog post content")
        assert result["score"] == 92.0

    def test_parses_feedback(self, db):
        w = BlogWriter(db)
        with patch.object(w, '_generate_with_llm', return_value=FAKE_GRADE_RESPONSE):
            result = w._grade_blog_post("some blog post content")
        assert "hook" in result["feedback"].lower()

    def test_defaults_when_parse_fails(self, db):
        w = BlogWriter(db)
        with patch.object(w, '_generate_with_llm', return_value="No structured response here."):
            result = w._grade_blog_post("content")
        assert result["grade"] == "C"
        assert result["score"] == 75.0


# ── Provider dispatch ─────────────────────────────────────────────────────────

class TestProviderDispatch:
    def _writer(self, db, provider):
        return BlogWriter(db, llm_provider=provider, llm_model="test-model")

    def test_ollama_provider_calls_ollama(self, db):
        w = self._writer(db, "ollama")
        with patch.object(w, '_generate_ollama', return_value=FAKE_POST) as mock:
            result = w._generate_with_llm("prompt")
        mock.assert_called_once_with("prompt")
        assert result == FAKE_POST

    def test_openai_provider_calls_openai(self, db):
        w = self._writer(db, "openai")
        with patch.object(w, '_generate_openai', return_value=FAKE_POST) as mock:
            result = w._generate_with_llm("prompt")
        mock.assert_called_once_with("prompt")

    def test_anthropic_provider_calls_anthropic(self, db):
        w = self._writer(db, "anthropic")
        with patch.object(w, '_generate_anthropic', return_value=FAKE_POST) as mock:
            result = w._generate_with_llm("prompt")
        mock.assert_called_once_with("prompt")

    def test_gemini_provider_calls_gemini(self, db):
        w = self._writer(db, "gemini")
        with patch.object(w, '_generate_gemini', return_value=FAKE_POST) as mock:
            result = w._generate_with_llm("prompt")
        mock.assert_called_once_with("prompt")

    def test_unknown_provider_returns_error_string(self, db):
        w = self._writer(db, "unknown-provider")
        result = w._generate_with_llm("prompt")
        assert "unsupported" in result.lower()


# ── API key loading ───────────────────────────────────────────────────────────

class TestApiKeyLoading:
    def test_openai_key_from_env(self, db, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
        w = BlogWriter(db, llm_provider="openai")
        assert w.api_key == "test-openai"

    def test_anthropic_key_from_env(self, db, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
        w = BlogWriter(db, llm_provider="anthropic")
        assert w.api_key == "test-anthropic"

    def test_gemini_key_from_env(self, db, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
        w = BlogWriter(db, llm_provider="gemini")
        assert w.api_key == "test-gemini"

    def test_ollama_has_no_api_key(self, db, monkeypatch):
        w = BlogWriter(db, llm_provider="ollama")
        assert w.api_key is None


# ── Blog post generation loop ─────────────────────────────────────────────────

class TestBlogPostGeneration:
    def test_returns_required_keys(self, db):
        w = BlogWriter(db, llm_provider="ollama", target_grade=80.0)
        with patch.object(w, '_generate_with_llm', side_effect=[
            FAKE_POST,                # initial draft
            FAKE_GRADE_RESPONSE,      # grade iteration 1  → score 92 ≥ 80, stops
        ]):
            result = w.generate_blog_post_from_digest("AI trends", SAMPLE_DIGEST)
        assert "final_post" in result
        assert "final_grade" in result
        assert "final_score" in result
        assert "iterations" in result
        assert "slug" in result
        assert "metadata" in result

    def test_stops_when_target_grade_met(self, db):
        w = BlogWriter(db, llm_provider="ollama", target_grade=80.0)
        # score 92 ≥ 80 on first grade → only 1 iteration
        with patch.object(w, '_generate_with_llm', side_effect=[
            FAKE_POST,
            FAKE_GRADE_RESPONSE,
        ]):
            result = w.generate_blog_post_from_digest("AI trends", SAMPLE_DIGEST)
        assert len(result["iterations"]) == 1

    def test_iterates_when_grade_not_met(self, db):
        low_grade = "GRADE: C\nSCORE: 70\nFEEDBACK: Needs improvement."
        w = BlogWriter(db, llm_provider="ollama", target_grade=90.0, )
        w.max_iterations = 2
        with patch.object(w, '_generate_with_llm', side_effect=[
            FAKE_POST,        # initial draft
            low_grade,        # grade 1 → 70 < 90, improve
            FAKE_POST,        # improved draft
            low_grade,        # grade 2 → max iterations hit
        ]):
            result = w.generate_blog_post_from_digest("AI trends", SAMPLE_DIGEST)
        assert len(result["iterations"]) == 2


# ── Blog post save ────────────────────────────────────────────────────────────

class TestSaveBlogPost:
    def test_saves_file_with_correct_name(self, db, tmp_path):
        w = BlogWriter(db)
        fake_result = {
            "topic": "AI trends",
            "slug": "ai-trends",
            "final_post": FAKE_POST,
            "final_grade": "A-",
            "final_score": 92.0,
            "iterations": [{"iteration": 1, "grade": "A-", "score": 92.0, "feedback": "Good."}],
            "metadata": {
                "episode_title": "Ep",
                "podcast_title": "Show",
                "generated_at": datetime.now().isoformat(),
                "model_used": "test-model",
            },
        }
        path = w.save_blog_post(fake_result, output_dir=str(tmp_path))
        assert path.endswith(".md")
        assert "ai-trends" in path
        assert (tmp_path / path.split("/")[-1]).exists()

    def test_saved_file_contains_frontmatter(self, db, tmp_path):
        w = BlogWriter(db)
        fake_result = {
            "topic": "AI trends",
            "slug": "ai-trends",
            "final_post": FAKE_POST,
            "final_grade": "A",
            "final_score": 95.0,
            "iterations": [],
            "metadata": {
                "episode_title": "Ep", "podcast_title": "Show",
                "generated_at": datetime.now().isoformat(), "model_used": "m",
            },
        }
        path = w.save_blog_post(fake_result, output_dir=str(tmp_path))
        content = open(path).read()
        assert "---" in content
        assert "final_grade" in content
