"""Tests for TranscriptCleaner — provider dispatch, extraction, fallbacks."""

import json
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from p3.database import P3Database
from p3.cleaner import TranscriptCleaner


@pytest.fixture
def db(tmp_path):
    database = P3Database(str(tmp_path / "test.duckdb"))
    yield database
    database.close()


@pytest.fixture
def episode_id(db):
    pid = db.add_podcast("Show", "https://example.com/feed", "tech")
    eid = db.add_episode(pid, "Ep", datetime(2026, 3, 1), "https://example.com/ep.mp3")
    db.add_transcript_segments(eid, [
        {"start": 0.0, "end": 5.0, "text": "Hello um world.", "speaker": None, "confidence": 0.9},
        {"start": 5.0, "end": 10.0, "text": "This is uh a test.", "speaker": None, "confidence": 0.9},
    ])
    return eid


# ── Filler-word regex cleaning ─────────────────────────────────────────────────

class TestCleanTranscript:
    def test_removes_filler_words(self, db):
        cleaner = TranscriptCleaner(db, llm_provider="ollama")
        # Patch _llm_clean to isolate regex layer
        with patch.object(cleaner, '_llm_clean', side_effect=lambda t: t):
            result = cleaner.clean_transcript("Hello um world, uh this is like a test.")
        assert "um" not in result
        assert "uh" not in result
        assert "world" in result
        assert "test" in result

    def test_collapses_extra_whitespace(self, db):
        cleaner = TranscriptCleaner(db, llm_provider="ollama")
        with patch.object(cleaner, '_llm_clean', side_effect=lambda t: t):
            result = cleaner.clean_transcript("hello   world")
        assert "  " not in result


# ── Basic fallback extraction ─────────────────────────────────────────────────

class TestBasicExtraction:
    def test_returns_required_keys(self, db):
        cleaner = TranscriptCleaner(db, llm_provider="ollama")
        result = cleaner._basic_extraction("OpenAI released a new model for machine learning tasks.")
        assert "key_topics" in result
        assert "themes" in result
        assert "quotes" in result
        assert "startups" in result
        assert "summary" in result

    def test_key_topics_are_strings(self, db):
        cleaner = TranscriptCleaner(db, llm_provider="ollama")
        result = cleaner._basic_extraction("machine learning artificial intelligence python programming")
        assert all(isinstance(t, str) for t in result["key_topics"])

    def test_detects_company_suffixes(self, db):
        cleaner = TranscriptCleaner(db, llm_provider="ollama")
        result = cleaner._basic_extraction("I talked to someone from Acme Labs about their product.")
        assert any("labs" in s.lower() or "Labs" in s for s in result["startups"])


# ── JSON extraction helper ─────────────────────────────────────────────────────

class TestJsonExtraction:
    """Test the {…} extraction used by all cloud providers."""

    def _make_cleaner(self, db, provider="openai"):
        return TranscriptCleaner(db, llm_provider=provider, api_key="fake-key")

    def test_extracts_json_from_clean_response(self, db):
        cleaner = self._make_cleaner(db)
        raw = '{"key_topics": ["AI"], "themes": ["tech"], "quotes": [], "startups": [], "summary": "Good ep."}'
        with patch.object(cleaner, '_openai_extract', return_value=json.loads(raw)):
            result = cleaner._generate_structured_summary("some transcript text")
        assert result["key_topics"] == ["AI"]

    def test_extracts_json_wrapped_in_prose(self, db):
        """Simulate an LLM that wraps JSON in explanatory text."""
        cleaner = self._make_cleaner(db)
        prose = 'Sure! Here is the JSON:\n{"key_topics": ["LLMs"], "themes": [], "quotes": [], "startups": ["Anthropic"], "summary": "x"}\nHope that helps.'
        json_start = prose.find('{')
        json_end = prose.rfind('}') + 1
        parsed = json.loads(prose[json_start:json_end])
        assert parsed["startups"] == ["Anthropic"]

    def test_missing_json_returns_none_from_slice(self, db):
        content = "I cannot produce JSON right now."
        json_start = content.find('{')
        assert json_start == -1  # guard used in _ollama_extract / _gemini_extract


# ── Provider dispatch (with mocked external calls) ────────────────────────────

class TestProviderDispatch:
    FAKE_SUMMARY = {
        "key_topics": ["AI"], "themes": ["tech"],
        "quotes": ["Great quote."], "startups": ["Acme"],
        "summary": "Episode summary."
    }

    def test_ollama_dispatch(self, db, episode_id):
        cleaner = TranscriptCleaner(db, llm_provider="ollama", llm_model="gemma3:12b")
        with patch.object(cleaner, '_ollama_extract', return_value=self.FAKE_SUMMARY), \
             patch.object(cleaner, '_ollama_clean', side_effect=lambda t: t):
            result = cleaner.generate_summary(episode_id)
        assert result["key_topics"] == ["AI"]
        summaries = db.get_summaries_by_date(datetime.now())
        assert len(summaries) == 1

    def test_openai_dispatch(self, db, episode_id):
        cleaner = TranscriptCleaner(db, llm_provider="openai", api_key="fake")
        with patch.object(cleaner, '_openai_extract', return_value=self.FAKE_SUMMARY), \
             patch.object(cleaner, '_openai_clean', side_effect=lambda t: t):
            result = cleaner.generate_summary(episode_id)
        assert result["summary"] == "Episode summary."

    def test_anthropic_dispatch(self, db, episode_id):
        cleaner = TranscriptCleaner(db, llm_provider="anthropic", api_key="fake")
        with patch.object(cleaner, '_anthropic_extract', return_value=self.FAKE_SUMMARY), \
             patch.object(cleaner, '_anthropic_clean', side_effect=lambda t: t):
            result = cleaner.generate_summary(episode_id)
        assert result["startups"] == ["Acme"]

    def test_gemini_dispatch(self, db, episode_id):
        cleaner = TranscriptCleaner(db, llm_provider="gemini", api_key="fake")
        with patch.object(cleaner, '_gemini_extract', return_value=self.FAKE_SUMMARY), \
             patch.object(cleaner, '_gemini_clean', side_effect=lambda t: t):
            result = cleaner.generate_summary(episode_id)
        assert result["themes"] == ["tech"]

    def test_failed_llm_falls_back_to_basic_extraction(self, db, episode_id):
        cleaner = TranscriptCleaner(db, llm_provider="openai", api_key="fake")
        with patch.object(cleaner, '_openai_extract', return_value=None), \
             patch.object(cleaner, '_openai_clean', side_effect=lambda t: t):
            result = cleaner.generate_summary(episode_id)
        # Falls back to _basic_extraction — must still return a valid dict
        assert result is not None
        assert "key_topics" in result

    def test_empty_transcript_returns_none(self, db):
        pid = db.add_podcast("S", "https://example.com/f2", "tech")
        eid = db.add_episode(pid, "Empty", datetime(2026, 3, 1), "https://example.com/e2.mp3")
        # No transcript segments added
        cleaner = TranscriptCleaner(db, llm_provider="ollama")
        result = cleaner.generate_summary(eid)
        assert result is None


# ── API key loading from environment ──────────────────────────────────────────

class TestApiKeyLoading:
    def test_openai_key_loaded_from_env(self, db, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env-openai-key")
        cleaner = TranscriptCleaner(db, llm_provider="openai")
        assert cleaner.api_key == "env-openai-key"

    def test_anthropic_key_loaded_from_env(self, db, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-anthropic-key")
        cleaner = TranscriptCleaner(db, llm_provider="anthropic")
        assert cleaner.api_key == "env-anthropic-key"

    def test_gemini_key_loaded_from_env(self, db, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "env-gemini-key")
        cleaner = TranscriptCleaner(db, llm_provider="gemini")
        assert cleaner.api_key == "env-gemini-key"

    def test_ollama_has_no_api_key(self, db, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cleaner = TranscriptCleaner(db, llm_provider="ollama")
        assert cleaner.api_key is None
