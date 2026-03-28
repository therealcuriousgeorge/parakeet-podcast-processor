"""Tests for the DuckDB storage layer."""

import pytest
from datetime import datetime
from p3.database import P3Database


@pytest.fixture
def db(tmp_path):
    """In-memory-equivalent DuckDB at a temp path, closed after each test."""
    database = P3Database(str(tmp_path / "test.duckdb"))
    yield database
    database.close()


# ── Podcasts ──────────────────────────────────────────────────────────────────

class TestPodcasts:
    def test_add_and_retrieve_podcast(self, db):
        pid = db.add_podcast("My Show", "https://example.com/feed", "tech")
        assert isinstance(pid, int)
        result = db.get_podcast_by_url("https://example.com/feed")
        assert result is not None
        assert result["title"] == "My Show"
        assert result["category"] == "tech"

    def test_get_nonexistent_podcast_returns_none(self, db):
        assert db.get_podcast_by_url("https://nope.example.com") is None

    def test_duplicate_podcast_url_returns_existing_on_reuse(self, db):
        id1 = db.add_podcast("Show A", "https://example.com/feed", "tech")
        # Inserting a duplicate URL would violate the UNIQUE constraint.
        # The downloader guards this with get_podcast_by_url first — test that pattern.
        existing = db.get_podcast_by_url("https://example.com/feed")
        assert existing["id"] == id1


# ── Episodes ──────────────────────────────────────────────────────────────────

class TestEpisodes:
    @pytest.fixture
    def podcast_id(self, db):
        return db.add_podcast("Test Show", "https://example.com/feed", "tech")

    def test_add_episode_and_check_exists(self, db, podcast_id):
        db.add_episode(podcast_id, "Episode 1", datetime(2026, 3, 1),
                       "https://example.com/ep1.mp3", "/data/audio/ep1.wav")
        assert db.episode_exists("https://example.com/ep1.mp3")

    def test_episode_does_not_exist(self, db):
        assert not db.episode_exists("https://nope.example.com/ep.mp3")

    def test_get_episodes_by_status(self, db, podcast_id):
        db.add_episode(podcast_id, "Ep A", datetime(2026, 3, 1),
                       "https://example.com/a.mp3", "/data/audio/a.wav")
        db.add_episode(podcast_id, "Ep B", datetime(2026, 3, 2),
                       "https://example.com/b.mp3", "/data/audio/b.wav")
        episodes = db.get_episodes_by_status("downloaded")
        assert len(episodes) == 2
        assert all(ep["status"] == "downloaded" for ep in episodes)

    def test_update_episode_status(self, db, podcast_id):
        eid = db.add_episode(podcast_id, "Ep X", datetime(2026, 3, 1),
                             "https://example.com/x.mp3", "/data/audio/x.wav")
        db.update_episode_status(eid, "transcribed")
        transcribed = db.get_episodes_by_status("transcribed")
        assert any(ep["id"] == eid for ep in transcribed)
        assert not any(ep["id"] == eid for ep in db.get_episodes_by_status("downloaded"))

    def test_episodes_have_podcast_title(self, db, podcast_id):
        db.add_episode(podcast_id, "Ep Y", datetime(2026, 3, 1),
                       "https://example.com/y.mp3")
        episodes = db.get_episodes_by_status("downloaded")
        assert episodes[0]["podcast_title"] == "Test Show"


# ── Transcripts ───────────────────────────────────────────────────────────────

class TestTranscripts:
    @pytest.fixture
    def episode_id(self, db):
        pid = db.add_podcast("Show", "https://example.com/feed", "tech")
        return db.add_episode(pid, "Ep", datetime(2026, 3, 1),
                              "https://example.com/ep.mp3")

    def test_add_and_retrieve_segments(self, db, episode_id):
        segments = [
            {"start": 0.0, "end": 5.0, "text": "Hello world.", "speaker": None, "confidence": 0.9},
            {"start": 5.0, "end": 10.0, "text": "How are you?", "speaker": None, "confidence": 0.85},
        ]
        db.add_transcript_segments(episode_id, segments)
        rows = db.get_transcripts_for_episode(episode_id)
        assert len(rows) == 2
        assert rows[0]["text"] == "Hello world."
        assert rows[1]["timestamp_start"] == 5.0

    def test_segments_ordered_by_start_time(self, db, episode_id):
        segments = [
            {"start": 10.0, "end": 15.0, "text": "Second.", "speaker": None, "confidence": 1.0},
            {"start": 0.0,  "end": 5.0,  "text": "First.",  "speaker": None, "confidence": 1.0},
        ]
        db.add_transcript_segments(episode_id, segments)
        rows = db.get_transcripts_for_episode(episode_id)
        assert rows[0]["text"] == "First."
        assert rows[1]["text"] == "Second."


# ── Summaries ─────────────────────────────────────────────────────────────────

class TestSummaries:
    @pytest.fixture
    def episode_id(self, db):
        pid = db.add_podcast("Show", "https://example.com/feed", "tech")
        return db.add_episode(pid, "Ep", datetime(2026, 3, 1),
                              "https://example.com/ep.mp3")

    def test_add_and_retrieve_summary(self, db, episode_id):
        target_date = datetime(2026, 3, 24)
        structured = '{"one_liner": "A great episode about AI.", "concepts_discussed": ["AI", "LLMs"]}'
        db.add_summary(
            episode_id=episode_id,
            key_topics=["AI", "LLMs"],
            themes=["automation"],
            quotes=["AI is the future."],
            startups=["OpenAI"],
            full_summary="A great episode about AI.",
            structured_summary=structured,
            digest_date=target_date,
        )
        summaries = db.get_summaries_by_date(target_date)
        assert len(summaries) == 1
        s = summaries[0]
        assert s["key_topics"] == ["AI", "LLMs"]
        assert s["themes"] == ["automation"]
        assert s["quotes"] == ["AI is the future."]
        assert s["startups"] == ["OpenAI"]
        assert s["full_summary"] == "A great episode about AI."
        assert s["structured_summary"] == structured
        assert s["episode_title"] == "Ep"
        assert s["podcast_title"] == "Show"

    def test_structured_summary_defaults_to_none(self, db, episode_id):
        db.add_summary(episode_id, [], [], [], [], "plain summary", datetime(2026, 3, 24))
        summaries = db.get_summaries_by_date(datetime(2026, 3, 24))
        assert summaries[0]["structured_summary"] is None

    def test_no_summaries_for_wrong_date(self, db, episode_id):
        db.add_summary(episode_id, ["topic"], [], [], [], "summary",
                       datetime(2026, 3, 24))
        assert db.get_summaries_by_date(datetime(2026, 3, 25)) == []

    def test_summary_defaults_digest_date_to_today(self, db, episode_id):
        from datetime import date
        db.add_summary(episode_id, [], [], [], [], "no date passed")
        today = datetime.now()
        summaries = db.get_summaries_by_date(today)
        assert len(summaries) == 1
