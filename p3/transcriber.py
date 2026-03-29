"""Audio transcription using Whisper, Parakeet, or OpenAI Whisper API."""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
import whisper

from .database import P3Database

try:
    from parakeet_mlx import from_pretrained as parakeet_from_pretrained
    PARAKEET_AVAILABLE = True
except ImportError:
    PARAKEET_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class AudioTranscriber:
    def __init__(self, db: P3Database, whisper_model: str = "base",
                 use_parakeet: bool = False,
                 parakeet_model: str = "mlx-community/parakeet-tdt-0.6b-v2",
                 transcription_provider: str = "local",
                 openai_api_key: str = None,
                 cleanup_audio: bool = False,
                 parakeet_chunk_duration: int = 600):
        self.db = db
        self.whisper_model = whisper_model
        self.use_parakeet = use_parakeet
        self.parakeet_model = parakeet_model
        self.transcription_provider = transcription_provider
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.cleanup_audio = cleanup_audio
        self.parakeet_chunk_duration = parakeet_chunk_duration
        self.whisper = None
        self.parakeet = None
        
    def _load_whisper(self):
        """Lazy load Whisper model."""
        if self.whisper is None:
            print(f"Loading Whisper model: {self.whisper_model}")
            self.whisper = whisper.load_model(self.whisper_model)

    def _load_parakeet(self):
        """Lazy load Parakeet MLX model."""
        if self.parakeet is None and PARAKEET_AVAILABLE:
            print(f"Loading Parakeet model: {self.parakeet_model}")
            try:
                self.parakeet = parakeet_from_pretrained(self.parakeet_model)
            except Exception as e:
                # MLX/Metal failures surface here as Python exceptions when
                # the GPU is unavailable or the model can't be loaded.
                raise RuntimeError(f"MLX/Parakeet model load failed: {e}") from e

    def transcribe_with_whisper(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio using OpenAI Whisper."""
        self._load_whisper()
        
        try:
            result = self.whisper.transcribe(
                audio_path,
                word_timestamps=True,
                verbose=False
            )
            
            # Convert Whisper output to our format
            segments = []
            for segment in result.get('segments', []):
                segments.append({
                    'start': segment.get('start', 0),
                    'end': segment.get('end', 0),
                    'text': segment.get('text', '').strip(),
                    'speaker': None,  # Whisper doesn't do speaker detection
                    'confidence': segment.get('no_speech_prob', 0.0)
                })
            
            return {
                'segments': segments,
                'language': result.get('language'),
                'text': result.get('text', ''),
                'provider': 'whisper'
            }
            
        except Exception as e:
            print(f"Whisper transcription failed: {e}")
            return None

    def transcribe_with_parakeet(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """Transcribe audio using Nvidia Parakeet MLX."""
        if not PARAKEET_AVAILABLE:
            print("Parakeet MLX not available, falling back to Whisper")
            return self.transcribe_with_whisper(audio_path)

        self._load_parakeet()

        try:
            # chunk_duration splits long files into GPU-safe segments.
            # Default 600s (10 min) keeps Metal memory well under 9.5 GB limit.
            result = self.parakeet.transcribe(
                audio_path,
                chunk_duration=self.parakeet_chunk_duration,
                overlap_duration=15.0,
            )

            segments = []
            for sentence in result.sentences:
                segments.append({
                    'start': sentence.start,
                    'end': sentence.end,
                    'text': sentence.text.strip(),
                    'speaker': None,
                    'confidence': 1.0
                })

            return {
                'segments': segments,
                'language': 'en',
                'text': result.text,
                'provider': 'parakeet-mlx'
            }

        except Exception as e:
            print(f"Parakeet transcription failed: {e}")
            print("Falling back to Whisper")
            return self.transcribe_with_whisper(audio_path)

    def transcribe_with_openai_api(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """Transcribe audio using the OpenAI Whisper cloud API."""
        if not OPENAI_AVAILABLE:
            print("openai package not installed. Run: pip install openai — falling back to local Whisper")
            return self.transcribe_with_whisper(audio_path)

        if not self.openai_api_key:
            print("OPENAI_API_KEY not set — falling back to local Whisper")
            return self.transcribe_with_whisper(audio_path)

        try:
            client = OpenAI(api_key=self.openai_api_key)
            with open(audio_path, 'rb') as audio_file:
                result = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )

            segments = []
            for segment in (result.segments or []):
                segments.append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip(),
                    'speaker': None,
                    'confidence': 1.0,
                })

            return {
                'segments': segments,
                'language': result.language,
                'text': result.text,
                'provider': 'openai-api',
            }

        except Exception as e:
            print(f"OpenAI Whisper API transcription failed: {e}")
            print("Falling back to local Whisper")
            return self.transcribe_with_whisper(audio_path)

    def transcribe_episode(self, episode_id: int) -> bool:
        """Transcribe a single episode and store results."""
        episodes = self.db.get_episodes_by_status('downloaded')
        episode = next((ep for ep in episodes if ep['id'] == episode_id), None)

        if not episode:
            print(f"Episode {episode_id} not found or not in 'downloaded' status")
            return False

        if not episode['file_path'] or not Path(episode['file_path']).exists():
            err = FileNotFoundError(f"Audio file missing: {episode.get('file_path')}")
            print(f"✗ {err}")
            self.db.add_error(episode_id, 'transcription', err)
            return False

        print(f"Transcribing: {episode['title']}")

        try:
            if self.transcription_provider == 'openai-api':
                result = self.transcribe_with_openai_api(episode['file_path'])
            elif self.transcription_provider == 'local-parakeet' or self.use_parakeet:
                result = self.transcribe_with_parakeet(episode['file_path'])
            else:
                result = self.transcribe_with_whisper(episode['file_path'])
        except Exception as e:
            print(f"✗ Transcription crashed for episode {episode_id}: {e}")
            self.db.add_error(episode_id, 'transcription', e)
            return False

        if not result:
            err = RuntimeError("Transcription returned no output — provider may have failed silently")
            print(f"✗ {err}")
            self.db.add_error(episode_id, 'transcription', err)
            return False

        try:
            self.db.add_transcript_segments(episode_id, result['segments'])
            self.db.update_episode_status(episode_id, 'transcribed')
        except Exception as e:
            print(f"✗ Failed to save transcript for episode {episode_id}: {e}")
            self.db.add_error(episode_id, 'transcription', e)
            return False

        if self.cleanup_audio:
            audio_path = Path(episode['file_path'])
            if audio_path.exists():
                audio_path.unlink()
                print(f"✓ Deleted audio: {audio_path.name}")

        print(f"✓ Transcribed: {episode['title']}")
        return True

    def transcribe_all_pending(self) -> int:
        """Transcribe all episodes with 'downloaded' status."""
        episodes = self.db.get_episodes_by_status('downloaded')
        transcribed_count = 0
        
        for episode in episodes:
            if self.transcribe_episode(episode['id']):
                transcribed_count += 1
            
        return transcribed_count

    def get_full_transcript(self, episode_id: int) -> str:
        """Get the full transcript text for an episode."""
        segments = self.db.get_transcripts_for_episode(episode_id)
        return "\n".join(segment['text'] for segment in segments)

    def export_transcript(self, episode_id: int, format: str = "txt") -> str:
        """Export transcript in various formats."""
        segments = self.db.get_transcripts_for_episode(episode_id)
        
        if format == "txt":
            return "\n".join(segment['text'] for segment in segments)
        
        elif format == "srt":
            srt_content = []
            for i, segment in enumerate(segments, 1):
                start_time = self._seconds_to_srt_time(segment['timestamp_start'] or 0)
                end_time = self._seconds_to_srt_time(segment['timestamp_end'] or 0)
                srt_content.append(f"{i}\n{start_time} --> {end_time}\n{segment['text']}\n")
            return "\n".join(srt_content)
        
        elif format == "json":
            return json.dumps(segments, indent=2, default=str)
        
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
