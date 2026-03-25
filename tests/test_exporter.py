"""Tests for DigestExporter — markdown, JSON, and HTML rendering."""

import json
import pytest
from datetime import date

from p3.exporter import DigestExporter


SAMPLE_SUMMARIES = [
    {
        "id": 1,
        "episode_id": 1,
        "podcast_title": "Tech Talk",
        "episode_title": "AI in 2026",
        "full_summary": "A wide-ranging discussion about the state of AI.",
        "key_topics": ["LLMs", "agents"],
        "themes": ["automation", "productivity"],
        "quotes": ["The future is already here."],
        "startups": ["OpenAI", "Anthropic"],
        "digest_date": date(2026, 3, 24),
        "created_at": "2026-03-24T10:00:00",
    },
    {
        "id": 2,
        "episode_id": 2,
        "podcast_title": "Tech Talk",
        "episode_title": "Cloud vs Local",
        "full_summary": "Comparing cloud and local AI deployments.",
        "key_topics": ["cloud", "on-premise"],
        "themes": ["cost", "privacy"],
        "quotes": [],
        "startups": [],
        "digest_date": date(2026, 3, 24),
        "created_at": "2026-03-24T11:00:00",
    },
]


@pytest.fixture
def exporter():
    return DigestExporter(db=None)


# ── Markdown export ───────────────────────────────────────────────────────────

class TestMarkdownExport:
    def test_contains_date_heading(self, exporter):
        md = exporter.export_markdown(SAMPLE_SUMMARIES, date(2026, 3, 24))
        assert "2026-03-24" in md

    def test_contains_podcast_name(self, exporter):
        md = exporter.export_markdown(SAMPLE_SUMMARIES, date(2026, 3, 24))
        assert "Tech Talk" in md

    def test_contains_episode_titles(self, exporter):
        md = exporter.export_markdown(SAMPLE_SUMMARIES, date(2026, 3, 24))
        assert "AI in 2026" in md
        assert "Cloud vs Local" in md

    def test_contains_summary_text(self, exporter):
        md = exporter.export_markdown(SAMPLE_SUMMARIES, date(2026, 3, 24))
        assert "wide-ranging discussion" in md

    def test_contains_key_topics(self, exporter):
        md = exporter.export_markdown(SAMPLE_SUMMARIES, date(2026, 3, 24))
        assert "LLMs" in md
        assert "agents" in md

    def test_contains_quotes_as_blockquote(self, exporter):
        md = exporter.export_markdown(SAMPLE_SUMMARIES, date(2026, 3, 24))
        assert "> The future is already here." in md

    def test_contains_startups(self, exporter):
        md = exporter.export_markdown(SAMPLE_SUMMARIES, date(2026, 3, 24))
        assert "OpenAI" in md
        assert "Anthropic" in md

    def test_empty_summaries_returns_gracefully(self, exporter):
        md = exporter.export_markdown([], date(2026, 3, 24))
        assert "No summaries" in md

    def test_episodes_grouped_under_podcast(self, exporter):
        md = exporter.export_markdown(SAMPLE_SUMMARIES, date(2026, 3, 24))
        tech_pos = md.find("Tech Talk")
        ep1_pos = md.find("AI in 2026")
        ep2_pos = md.find("Cloud vs Local")
        # Both episodes appear after the podcast heading
        assert tech_pos < ep1_pos
        assert tech_pos < ep2_pos

    def test_multiple_podcasts_get_separate_sections(self, exporter):
        multi = SAMPLE_SUMMARIES + [{
            **SAMPLE_SUMMARIES[0],
            "id": 3,
            "podcast_title": "Other Show",
            "episode_title": "Different Ep",
        }]
        md = exporter.export_markdown(multi, date(2026, 3, 24))
        assert "Tech Talk" in md
        assert "Other Show" in md


# ── JSON export ───────────────────────────────────────────────────────────────

class TestJsonExport:
    def test_valid_json(self, exporter):
        raw = exporter.export_json(SAMPLE_SUMMARIES, date(2026, 3, 24))
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_contains_date_and_count(self, exporter):
        raw = exporter.export_json(SAMPLE_SUMMARIES, date(2026, 3, 24))
        parsed = json.loads(raw)
        assert parsed["date"] == "2026-03-24"
        assert parsed["total_episodes"] == 2

    def test_summaries_list_has_all_entries(self, exporter):
        raw = exporter.export_json(SAMPLE_SUMMARIES, date(2026, 3, 24))
        parsed = json.loads(raw)
        assert len(parsed["summaries"]) == 2

    def test_empty_summaries_json(self, exporter):
        raw = exporter.export_json([], date(2026, 3, 24))
        parsed = json.loads(raw)
        assert parsed["total_episodes"] == 0
        assert parsed["summaries"] == []


# ── HTML export ───────────────────────────────────────────────────────────────

class TestHtmlExport:
    def test_is_valid_html_structure(self, exporter):
        html = exporter.export_email_html(SAMPLE_SUMMARIES, date(2026, 3, 24))
        assert "<html>" in html
        assert "</html>" in html
        assert "<body>" in html

    def test_contains_episode_content(self, exporter):
        html = exporter.export_email_html(SAMPLE_SUMMARIES, date(2026, 3, 24))
        assert "AI in 2026" in html
        assert "LLMs" in html

    def test_empty_summaries_html(self, exporter):
        html = exporter.export_email_html([], date(2026, 3, 24))
        assert "No summaries" in html
