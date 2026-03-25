"""Tests for AudioTranscriber — provider routing, SRT export, fallbacks."""

import pytest
from unittest.mock import MagicMock, patch, mock_open
from datetime import datetime
from pathlib import Path

from p3.database import P3Database
from p3.transcriber import AudioTranscriber


@pytest.fixture
def db(tmp_path):
    database = P3Database(str(tmp_path / "test.duckdb"))
    yield database
    database.close()


@pytest.fixture
def episode_id(db, tmp_path):
    """Episode with a real (empty) audio file so path checks pass."""
    audio_file = tmp_path / "ep.wav"
    audio_file.write_bytes(b"")
    pid = db.add_podcast("Show", "https://example.com/feed", "tech")
    eid = db.add_episode(pid, "Ep", datetime(2026, 3, 1),
                         "https://example.com/ep.mp3", str(audio_file))
    return eid


FAKE_RESULT = {
    "segments": [
        {"start": 0.0, "end": 5.0, "text": "Hello world.", "speaker": None, "confidence": 1.0},
        {"start": 5.0, "end": 10.0, "text": "This is a test.", "speaker": None, "confidence": 1.0},
    ],
    "language": "en",
    "text": "Hello world. This is a test.",
    "provider": "mock",
}


# ── Provider routing ──────────────────────────────────────────────────────────

class TestTranscriptionProviderRouting:
    def test_routes_to_parakeet_when_provider_is_local_parakeet(self, db, episode_id):
        t = AudioTranscriber(db, transcription_provider="local-parakeet", use_parakeet=True)
        with patch.object(t, 'transcribe_with_parakeet', return_value=FAKE_RESULT) as mock_p:
            t.transcribe_episode(episode_id)
        mock_p.assert_called_once()

    def test_routes_to_whisper_when_provider_is_local_whisper(self, db, episode_id):
        t = AudioTranscriber(db, transcription_provider="local-whisper")
        with patch.object(t, 'transcribe_with_whisper', return_value=FAKE_RESULT) as mock_w:
            t.transcribe_episode(episode_id)
        mock_w.assert_called_once()

    def test_routes_to_openai_api_when_provider_is_openai_api(self, db, episode_id):
        t = AudioTranscriber(db, transcription_provider="openai-api", openai_api_key="fake")
        with patch.object(t, 'transcribe_with_openai_api', return_value=FAKE_RESULT) as mock_o:
            t.transcribe_episode(episode_id)
        mock_o.assert_called_once()

    def test_legacy_use_parakeet_flag_still_works(self, db, episode_id):
        """Old configs using parakeet_enabled=True (no transcription_provider) must still route correctly."""
        t = AudioTranscriber(db, use_parakeet=True, transcription_provider="local")
        with patch.object(t, 'transcribe_with_parakeet', return_value=FAKE_RESULT) as mock_p:
            t.transcribe_episode(episode_id)
        mock_p.assert_called_once()


# ── Episode status after transcription ───────────────────────────────────────

class TestStatusTransition:
    def test_episode_status_set_to_transcribed_on_success(self, db, episode_id):
        t = AudioTranscriber(db, transcription_provider="local-whisper")
        with patch.object(t, 'transcribe_with_whisper', return_value=FAKE_RESULT):
            success = t.transcribe_episode(episode_id)
        assert success is True
        transcribed = db.get_episodes_by_status("transcribed")
        assert any(ep["id"] == episode_id for ep in transcribed)

    def test_failed_transcription_returns_false(self, db, episode_id):
        t = AudioTranscriber(db, transcription_provider="local-whisper")
        with patch.object(t, 'transcribe_with_whisper', return_value=None):
            success = t.transcribe_episode(episode_id)
        assert success is False
        # Episode stays at 'downloaded'
        downloaded = db.get_episodes_by_status("downloaded")
        assert any(ep["id"] == episode_id for ep in downloaded)

    def test_missing_audio_file_returns_false(self, db):
        pid = db.add_podcast("S", "https://example.com/f2", "tech")
        eid = db.add_episode(pid, "Ep", datetime(2026, 3, 1),
                             "https://example.com/e2.mp3", "/nonexistent/path.wav")
        t = AudioTranscriber(db, transcription_provider="local-whisper")
        assert t.transcribe_episode(eid) is False

    def test_transcribe_episode_not_in_downloaded_returns_false(self, db):
        t = AudioTranscriber(db, transcription_provider="local-whisper")
        assert t.transcribe_episode(9999) is False


# ── OpenAI API fallback ───────────────────────────────────────────────────────

class TestOpenAIAPIFallback:
    def test_falls_back_to_whisper_when_no_api_key(self, db, episode_id):
        t = AudioTranscriber(db, transcription_provider="openai-api", openai_api_key=None)
        with patch.object(t, 'transcribe_with_whisper', return_value=FAKE_RESULT) as mock_w, \
             patch.dict('os.environ', {}, clear=True):
            # Ensure OPENAI_API_KEY is absent
            import os
            os.environ.pop("OPENAI_API_KEY", None)
            t.openai_api_key = None
            t.transcribe_with_openai_api("/fake/path.wav")
        mock_w.assert_called_once()


# ── Audio cleanup after transcription ────────────────────────────────────────

class TestAudioCleanup:
    def test_audio_deleted_when_cleanup_enabled(self, db, tmp_path):
        audio_file = tmp_path / "ep.wav"
        audio_file.write_bytes(b"fake audio")
        pid = db.add_podcast("Show", "https://example.com/feed-c1", "tech")
        eid = db.add_episode(pid, "Ep", datetime(2026, 3, 1),
                             "https://example.com/c1.mp3", str(audio_file))

        t = AudioTranscriber(db, transcription_provider="local-whisper", cleanup_audio=True)
        with patch.object(t, 'transcribe_with_whisper', return_value=FAKE_RESULT):
            t.transcribe_episode(eid)

        assert not audio_file.exists()

    def test_audio_kept_when_cleanup_disabled(self, db, tmp_path):
        audio_file = tmp_path / "ep.wav"
        audio_file.write_bytes(b"fake audio")
        pid = db.add_podcast("Show", "https://example.com/feed-c2", "tech")
        eid = db.add_episode(pid, "Ep", datetime(2026, 3, 1),
                             "https://example.com/c2.mp3", str(audio_file))

        t = AudioTranscriber(db, transcription_provider="local-whisper", cleanup_audio=False)
        with patch.object(t, 'transcribe_with_whisper', return_value=FAKE_RESULT):
            t.transcribe_episode(eid)

        assert audio_file.exists()

    def test_audio_not_deleted_when_transcription_fails(self, db, tmp_path):
        audio_file = tmp_path / "ep.wav"
        audio_file.write_bytes(b"fake audio")
        pid = db.add_podcast("Show", "https://example.com/feed-c3", "tech")
        eid = db.add_episode(pid, "Ep", datetime(2026, 3, 1),
                             "https://example.com/c3.mp3", str(audio_file))

        t = AudioTranscriber(db, transcription_provider="local-whisper", cleanup_audio=True)
        with patch.object(t, 'transcribe_with_whisper', return_value=None):
            t.transcribe_episode(eid)

        # Transcription failed — audio must be preserved so user can retry
        assert audio_file.exists()


# ── Transcript export formats ─────────────────────────────────────────────────

class TestTranscriptExport:
    @pytest.fixture
    def transcriber_with_segments(self, db):
        pid = db.add_podcast("Show", "https://example.com/feed3", "tech")
        eid = db.add_episode(pid, "Ep", datetime(2026, 3, 1), "https://example.com/ep3.mp3")
        db.add_transcript_segments(eid, [
            {"start": 0.0,  "end": 5.0,  "text": "Hello.",   "speaker": None, "confidence": 1.0},
            {"start": 5.0,  "end": 10.0, "text": "Goodbye.", "speaker": None, "confidence": 1.0},
        ])
        return AudioTranscriber(db), eid

    def test_export_txt(self, transcriber_with_segments):
        t, eid = transcriber_with_segments
        txt = t.export_transcript(eid, "txt")
        assert "Hello." in txt
        assert "Goodbye." in txt

    def test_export_srt_contains_timestamps(self, transcriber_with_segments):
        t, eid = transcriber_with_segments
        srt = t.export_transcript(eid, "srt")
        assert "-->" in srt
        assert "Hello." in srt

    def test_export_json_is_valid(self, transcriber_with_segments):
        import json
        t, eid = transcriber_with_segments
        raw = t.export_transcript(eid, "json")
        parsed = json.loads(raw)
        assert isinstance(parsed, list)
        assert parsed[0]["text"] == "Hello."

    def test_export_unsupported_format_raises(self, transcriber_with_segments):
        t, eid = transcriber_with_segments
        with pytest.raises(ValueError):
            t.export_transcript(eid, "pdf")

    def test_get_full_transcript_joins_segments(self, transcriber_with_segments):
        t, eid = transcriber_with_segments
        text = t.get_full_transcript(eid)
        assert "Hello." in text
        assert "Goodbye." in text

    def test_srt_time_conversion(self, db):
        t = AudioTranscriber(db)
        assert t._seconds_to_srt_time(0) == "00:00:00,000"
        assert t._seconds_to_srt_time(3661.5) == "01:01:01,500"
