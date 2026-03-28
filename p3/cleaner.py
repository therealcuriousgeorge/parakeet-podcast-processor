"""LLM-based transcript cleaning and summarization."""

import json
import re
from datetime import datetime, date
from typing import Dict, List, Optional, Any
import httpx

SUMMARY_SYSTEM_PROMPT = """You are a summarization engine for a senior product leader with 17+ years of experience across Amazon, Walmart, Zillow, and other tech companies. He currently leads product management in Last Mile Delivery at Walmart Global Tech, working at the intersection of data, operations, supply chain optimization, and platform-scale systems. He thinks in frameworks and mental models, and values structured knowledge he can scan quickly and drill into selectively.

His core interests:
- Product management craft and leadership
- AI/ML and its applications to product, operations, and strategy
- Business strategy and competitive analysis (he follows Ben Thompson's Aggregation Theory lens closely)
- Supply chain, logistics, and marketplace/platform dynamics
- Technology transitions and how they reshape industries and professional roles
- Data-driven decision making and experimentation

Analyze the provided content (podcast transcript or newsletter) and return a JSON object with exactly these keys:

{
  "one_liner": "Single sentence capturing the core thesis — helps decide in 5 seconds whether to read further.",
  "concepts_discussed": ["short label 3-6 words", "..."],
  "key_concepts": [
    {
      "name": "Short memorable label 3-6 words",
      "summary": "2-3 sentences explaining the idea.",
      "why_it_matters": "One sentence connecting to product management, business strategy, or technology leadership."
    }
  ],
  "mental_models": [
    {
      "name": "Model or framework name",
      "how_it_works": "Brief explanation of the model's logic or structure.",
      "application": "How this model can be applied to product decisions, business strategy, or operational thinking."
    }
  ],
  "quotable_lines": [
    {
      "quote": "Exact quote or close paraphrase",
      "speaker": "Speaker name or null",
      "context": "Brief note on why it resonates or what it crystallizes."
    }
  ],
  "career_relevance": [
    "Specific bullet connecting content to PM leadership, platform thinking, supply chain, data/AI strategy, or competitive positioning."
  ],
  "verdict": {
    "novelty": 4,
    "actionability": 3,
    "depth": "Deep read",
    "best_sections": "Description of specific parts worth revisiting, or null."
  }
}

Guidelines:
- concepts_discussed: ordered by prominence, not alphabetically. Use as a scannable table of contents.
- key_concepts: 3-7 concepts depending on content density.
- mental_models: extract explicit or implicit frameworks. Map to known frameworks (Aggregation Theory, Jobs to Be Done, Wardley Mapping, disruption theory, network effects, platform dynamics) where applicable. Use [] if none are present — do not force-fit.
- quotable_lines: prioritize pithy, contrarian, or insight-crystallizing lines. Aim for 2-5.
- career_relevance: be specific — not "relevant to PMs" but how and where it applies.
- verdict.novelty: 1-5 (1=rehash of known ideas, 5=genuinely new thinking).
- verdict.actionability: 1-5 (1=purely abstract, 5=directly applicable to current work).
- verdict.depth: one of "Skim" | "Read key sections" | "Deep read" | "Reference material — save for later".
- Be concise. Lead with the insight, not narration. Preserve original speaker terminology for technical concepts.
- If content is thin or repetitive, say so honestly in the verdict. Do not inflate.
- Return valid JSON only. No markdown fences, no explanatory text outside the JSON object."""

from .database import P3Database

# Optional Ollama support
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Optional Anthropic support
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Optional Gemini support
try:
    from google import genai as google_genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class TranscriptCleaner:
    def __init__(self, db: P3Database, llm_provider: str = "openai", 
                 llm_model: str = "gpt-3.5-turbo", api_key: str = None, ollama_base_url: str = "http://localhost:11434"):
        self.db = db
        self.llm_provider = llm_provider.lower()
        self.llm_model = llm_model
        self.api_key = api_key
        self.ollama_base_url = ollama_base_url
        
        # Load API key from environment if not provided
        if not self.api_key and self.llm_provider != "ollama":
            import os
            key_map = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "gemini": "GEMINI_API_KEY",
            }
            self.api_key = os.getenv(key_map.get(self.llm_provider, ""))

    def clean_transcript(self, raw_text: str) -> str:
        """Clean transcript by removing filler words and improving readability."""
        # Basic cleaning without LLM first
        text = raw_text
        
        # Remove common filler words
        fillers = [
            r'\b(um|uh|ah|er|hmm|like|you know|sort of|kind of)\b',
            r'\b(actually|basically|literally|obviously|definitely)\b(?=\s+\w)',
        ]
        
        for filler in fillers:
            text = re.sub(filler, '', text, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # If we have API access or Ollama, use LLM for advanced cleaning
        if self.api_key or self.llm_provider == "ollama":
            try:
                text = self._llm_clean(text)
            except Exception as e:
                print(f"LLM cleaning failed, using basic cleaning: {e}")
        
        return text

    def _llm_clean(self, text: str) -> str:
        """Use LLM to clean and improve transcript."""
        if self.llm_provider == "openai":
            return self._openai_clean(text)
        elif self.llm_provider == "anthropic":
            return self._anthropic_clean(text)
        elif self.llm_provider == "ollama":
            return self._ollama_clean(text)
        elif self.llm_provider == "gemini":
            return self._gemini_clean(text)
        else:
            print(f"Unsupported LLM provider: {self.llm_provider}")
            return text

    def _openai_clean(self, text: str) -> str:
        """Clean transcript using OpenAI API."""
        prompt = """Clean this podcast transcript by:
1. Removing filler words (um, uh, like, you know)
2. Fixing grammar and punctuation
3. Preserving technical terms and proper nouns exactly
4. Maintaining the speaker's voice and meaning
5. Breaking into clear paragraphs

Return only the cleaned text, no additional commentary.

Transcript:
"""
        
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.llm_model,
                    "messages": [
                        {"role": "system", "content": "You are an expert transcript editor."},
                        {"role": "user", "content": prompt + text}
                    ],
                    "temperature": 0.1,
                    "max_tokens": min(len(text) * 2, 4000)
                }
            )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            raise Exception(f"OpenAI API error: {response.status_code}")

    def _anthropic_clean(self, text: str) -> str:
        """Clean transcript using Anthropic Claude API."""
        if not ANTHROPIC_AVAILABLE:
            print("anthropic package not installed. Run: pip install anthropic")
            return text

        prompt = """Clean this podcast transcript by:
1. Removing filler words (um, uh, like, you know)
2. Fixing grammar and punctuation
3. Preserving technical terms and proper nouns exactly
4. Maintaining the speaker's voice and meaning
5. Breaking into clear paragraphs

Return only the cleaned text, no additional commentary.

Transcript:
""" + text

        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            message = client.messages.create(
                model=self.llm_model,
                max_tokens=min(len(text) * 2, 4096),
                system="You are an expert transcript editor.",
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text.strip()
        except Exception as e:
            print(f"Anthropic cleaning failed: {e}")
            return text

    def _ollama_clean(self, text: str) -> str:
        """Clean transcript using Ollama local LLM."""
        if not OLLAMA_AVAILABLE:
            print("Ollama not available, skipping LLM cleaning")
            return text
            
        prompt = """Clean this podcast transcript by:
1. Removing filler words (um, uh, like, you know)
2. Fixing grammar and punctuation
3. Preserving technical terms and proper nouns exactly
4. Maintaining the speaker's voice and meaning
5. Breaking into clear paragraphs

Return only the cleaned text, no additional commentary.

Transcript:
""" + text

        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert transcript editor."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response['message']['content'].strip()
        except Exception as e:
            print(f"Ollama cleaning failed: {e}")
            return text

    def generate_summary(self, episode_id: int) -> Dict[str, Any]:
        """Generate structured summary of an episode."""
        try:
            segments = self.db.get_transcripts_for_episode(episode_id)
            full_text = "\n".join(segment['text'] for segment in segments)

            if not full_text.strip():
                return None

            cleaned_text = self.clean_transcript(full_text)
            summary_data = self._generate_structured_summary(cleaned_text)

            if summary_data:
                quotes = [
                    q.get('quote', '') for q in summary_data.get('quotable_lines', [])
                    if isinstance(q, dict)
                ]
                self.db.add_summary(
                    episode_id=episode_id,
                    key_topics=summary_data.get('concepts_discussed', []),
                    themes=[],
                    quotes=quotes,
                    startups=[],
                    full_summary=summary_data.get('one_liner', ''),
                    structured_summary=json.dumps(summary_data),
                    digest_date=datetime.now()
                )
                self.db.update_episode_status(episode_id, 'processed')

            return summary_data

        except Exception as e:
            print(f"✗ Digest failed for episode {episode_id}: {e}")
            self.db.add_error(episode_id, 'digest', e)
            return None

    def _generate_structured_summary(self, text: str) -> Optional[Dict[str, Any]]:
        """Generate structured summary using LLM."""
        if not self.api_key and self.llm_provider != "ollama":
            return self._basic_extraction(text)

        try:
            if self.llm_provider == "openai":
                result = self._openai_extract(text)
            elif self.llm_provider == "anthropic":
                result = self._anthropic_extract(text)
            elif self.llm_provider == "ollama":
                result = self._ollama_extract(text)
            elif self.llm_provider == "gemini":
                result = self._gemini_extract(text)
            else:
                result = None
        except Exception as e:
            print(f"LLM summarization failed: {e}")
            result = None

        return result if result is not None else self._basic_extraction(text)

    def _openai_extract(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract structured summary using OpenAI API."""
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.llm_model,
                    "messages": [
                        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Transcript:\n{text}"}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 2000,
                }
            )

        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"].strip()
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(content[json_start:json_end])

        return None

    def _anthropic_extract(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract structured summary using Anthropic Claude API."""
        if not ANTHROPIC_AVAILABLE:
            print("anthropic package not installed. Run: pip install anthropic")
            return None

        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            message = client.messages.create(
                model=self.llm_model,
                max_tokens=2000,
                system=SUMMARY_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"Transcript:\n{text}"}]
            )
            content = message.content[0].text.strip()
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(content[json_start:json_end])
            return None
        except Exception as e:
            print(f"Anthropic extraction failed: {e}")
            return None

    def _ollama_extract(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract structured summary using Ollama local LLM."""
        if not OLLAMA_AVAILABLE:
            print("Ollama not available, using basic extraction")
            return None

        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Transcript:\n{text}"}
                ]
            )
            content = response['message']['content'].strip()
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(content[json_start:json_end])
            return None
        except Exception as e:
            print(f"Ollama extraction failed: {e}")
            return None

    def _gemini_clean(self, text: str) -> str:
        """Clean transcript using Google Gemini API."""
        if not GEMINI_AVAILABLE:
            print("google-genai package not installed. Run: pip install google-genai")
            return text

        prompt = """Clean this podcast transcript by:
1. Removing filler words (um, uh, like, you know)
2. Fixing grammar and punctuation
3. Preserving technical terms and proper nouns exactly
4. Maintaining the speaker's voice and meaning
5. Breaking into clear paragraphs

Return only the cleaned text, no additional commentary.

Transcript:
""" + text

        try:
            client = google_genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=self.llm_model,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            print(f"Gemini cleaning failed: {e}")
            return text

    def _gemini_extract(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract structured summary using Google Gemini API."""
        if not GEMINI_AVAILABLE:
            print("google-genai package not installed. Run: pip install google-genai")
            return None

        try:
            client = google_genai.Client(api_key=self.api_key)
            full_prompt = f"{SUMMARY_SYSTEM_PROMPT}\n\nTranscript:\n{text}"
            response = client.models.generate_content(
                model=self.llm_model,
                contents=full_prompt,
            )
            content = response.text.strip()
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(content[json_start:json_end])
            return None
        except Exception as e:
            print(f"Gemini extraction failed: {e}")
            return None

    def _basic_extraction(self, text: str) -> Dict[str, Any]:
        """Basic keyword extraction as fallback when no LLM is available."""
        words = text.lower().split()
        word_freq: Dict[str, int] = {}
        for word in words:
            if len(word) > 4 and word.isalpha():
                word_freq[word] = word_freq.get(word, 0) + 1

        top_words = [
            word for word, _ in
            sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

        return {
            "one_liner": "Podcast episode — LLM summarization unavailable.",
            "concepts_discussed": top_words,
            "key_concepts": [],
            "mental_models": [],
            "quotable_lines": [],
            "career_relevance": [],
            "verdict": {
                "novelty": None,
                "actionability": None,
                "depth": "Skim",
                "best_sections": None,
            },
        }

    def process_all_transcribed(self) -> int:
        """Process all episodes with 'transcribed' status."""
        episodes = self.db.get_episodes_by_status('transcribed')
        processed_count = 0
        
        for episode in episodes:
            print(f"Processing summary for: {episode['title']}")
            if self.generate_summary(episode['id']):
                processed_count += 1
                print(f"✓ Processed: {episode['title']}")
            else:
                print(f"✗ Failed to process: {episode['title']}")
        
        return processed_count
