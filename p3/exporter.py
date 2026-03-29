"""Export functionality for P³ digests."""

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


def _slugify(text: str, max_len: int = 60) -> str:
    """Convert text to a safe filename slug."""
    slug = re.sub(r'[^\w\s-]', '', text)
    slug = re.sub(r'[\s_]+', '-', slug).strip('-')
    return slug[:max_len].rstrip('-')


def write_transcript(db, episode_id: int, output_dir: str = "transcripts") -> Optional[str]:
    """Write the full transcript for an episode to a markdown file.

    Returns the file path written, or None if no transcript segments exist.
    File name: YYYY-MM-DD_PodcastName_EpisodeTitle_transcript.md
    """
    episode = db.get_episode_by_id(episode_id)
    if not episode:
        return None

    segments = db.get_transcripts_for_episode(episode_id)
    if not segments:
        return None

    ep_date = str(episode.get('date', ''))[:10] or datetime.now().strftime('%Y-%m-%d')
    podcast_slug = _slugify(episode.get('podcast_title', 'podcast'))
    episode_slug = _slugify(episode.get('title', 'episode'))
    filename = f"{ep_date}_{podcast_slug}_{episode_slug}_transcript.md"

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / filename

    lines = [
        f"# {episode['podcast_title']}",
        f"## {episode['title']}",
        f"*{ep_date}*",
        "",
        "---",
        "",
    ]
    for seg in segments:
        start = seg.get('timestamp_start') or 0
        h, m, s = int(start // 3600), int((start % 3600) // 60), int(start % 60)
        timestamp = f"[{h:02d}:{m:02d}:{s:02d}]"
        lines.append(f"{timestamp} {seg['text']}")

    filepath.write_text("\n".join(lines))
    return str(filepath)


class DigestExporter:
    def __init__(self, db):
        self.db = db

    def _parse_structured(self, episode: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return parsed structured_summary dict, or None if unavailable."""
        raw = episode.get('structured_summary')
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    def _render_structured_md(self, s: Dict[str, Any]) -> List[str]:
        """Render a 7-section structured summary as Markdown lines."""
        lines: List[str] = []

        if s.get('one_liner'):
            lines += [f"**{s['one_liner']}**", ""]

        if s.get('concepts_discussed'):
            lines.append("**Concepts Discussed:**")
            for c in s['concepts_discussed']:
                lines.append(f"- {c}")
            lines.append("")

        if s.get('key_concepts'):
            lines.append("**Key Concepts:**")
            for kc in s['key_concepts']:
                lines.append(f"\n**{kc.get('name', '')}**")
                if kc.get('summary'):
                    lines.append(kc['summary'])
                if kc.get('why_it_matters'):
                    lines.append(f"*Why it matters:* {kc['why_it_matters']}")
            lines.append("")

        if s.get('mental_models'):
            lines.append("**Mental Models & Frameworks:**")
            for mm in s['mental_models']:
                lines.append(f"\n**{mm.get('name', '')}**")
                if mm.get('how_it_works'):
                    lines.append(mm['how_it_works'])
                if mm.get('application'):
                    lines.append(f"*Application:* {mm['application']}")
            lines.append("")

        if s.get('quotable_lines'):
            lines.append("**Quotable Lines:**")
            for ql in s['quotable_lines']:
                speaker = f" — {ql['speaker']}" if ql.get('speaker') else ""
                lines.append(f"> \"{ql.get('quote', '')}\"{speaker}")
                if ql.get('context'):
                    lines.append(f"> *{ql['context']}*")
            lines.append("")

        if s.get('career_relevance'):
            lines.append("**Career & Work Relevance:**")
            for cr in s['career_relevance']:
                lines.append(f"- {cr}")
            lines.append("")

        verdict = s.get('verdict')
        if verdict:
            novelty = verdict.get('novelty', '—')
            actionability = verdict.get('actionability', '—')
            depth = verdict.get('depth', '—')
            best = verdict.get('best_sections')
            lines.append(
                f"**Verdict:** Novelty {novelty}/5 · Actionability {actionability}/5 · {depth}"
            )
            if best:
                lines.append(f"*Best sections to revisit:* {best}")
            lines.append("")

        return lines

    def episode_filename(self, episode: Dict[str, Any], target_date: date, ext: str = "md") -> str:
        """Build a per-episode filename: YYYY-MM-DD_PodcastName_EpisodeTitle.ext"""
        date_str = str(target_date)
        podcast_slug = _slugify(episode.get('podcast_title', 'podcast'))
        episode_slug = _slugify(episode.get('episode_title', 'episode'))
        return f"{date_str}_{podcast_slug}_{episode_slug}.{ext}"

    def export_episode_markdown(self, episode: Dict[str, Any], target_date: date) -> str:
        """Render a single episode as a standalone Markdown document."""
        lines = [
            f"# {episode['podcast_title']}",
            f"## {episode['episode_title']}",
            f"*{target_date}*",
            "",
        ]
        structured = self._parse_structured(episode)
        if structured:
            lines.extend(self._render_structured_md(structured))
        else:
            if episode.get('full_summary'):
                lines.append(f"**Summary:** {episode['full_summary']}\n")
            if episode.get('key_topics'):
                lines.append("**Key Topics:**")
                for topic in episode['key_topics']:
                    lines.append(f"- {topic}")
                lines.append("")
            if episode.get('quotes'):
                lines.append("**Notable Quotes:**")
                for quote in episode['quotes']:
                    lines.append(f"> {quote}")
                lines.append("")
            if episode.get('startups'):
                lines.append("**Companies/Startups Mentioned:**")
                for startup in episode['startups']:
                    lines.append(f"- {startup}")
                lines.append("")
        return "\n".join(lines)

    def export_markdown(self, summaries: List[Dict[str, Any]], target_date: date) -> str:
        """Export summaries as Markdown."""
        content = [f"# Podcast Digest - {target_date}\n"]

        if not summaries:
            content.append("No summaries available for this date.\n")
            return "\n".join(content)

        by_podcast: Dict[str, List] = {}
        for summary in summaries:
            podcast = summary['podcast_title']
            by_podcast.setdefault(podcast, []).append(summary)

        for podcast_name, episodes in by_podcast.items():
            content.append(f"## {podcast_name}\n")
            for episode in episodes:
                content.append(f"### {episode['episode_title']}\n")
                structured = self._parse_structured(episode)
                if structured:
                    content.extend(self._render_structured_md(structured))
                else:
                    # Legacy fallback for summaries without structured data
                    if episode.get('full_summary'):
                        content.append(f"**Summary:** {episode['full_summary']}\n")
                    if episode.get('key_topics'):
                        content.append("**Key Topics:**")
                        for topic in episode['key_topics']:
                            content.append(f"- {topic}")
                        content.append("")
                    if episode.get('quotes'):
                        content.append("**Notable Quotes:**")
                        for quote in episode['quotes']:
                            content.append(f"> {quote}")
                        content.append("")
                    if episode.get('startups'):
                        content.append("**Companies/Startups Mentioned:**")
                        for startup in episode['startups']:
                            content.append(f"- {startup}")
                        content.append("")
                content.append("---\n")

        return "\n".join(content)

    def export_json(self, summaries: List[Dict[str, Any]], target_date: date) -> str:
        """Export summaries as JSON."""
        export_data = {
            "date": str(target_date),
            "total_episodes": len(summaries),
            "summaries": summaries
        }
        
        return json.dumps(export_data, indent=2, default=str)

    def _render_structured_html(self, s: Dict[str, Any]) -> str:
        """Render a 7-section structured summary as an HTML fragment."""
        html = ""

        if s.get('one_liner'):
            html += f'<p class="one-liner"><strong>{s["one_liner"]}</strong></p>'

        if s.get('concepts_discussed'):
            html += '<div class="concepts"><strong>Concepts Discussed:</strong><ul>'
            for c in s['concepts_discussed']:
                html += f'<li>{c}</li>'
            html += '</ul></div>'

        if s.get('key_concepts'):
            html += '<div class="key-concepts"><strong>Key Concepts:</strong>'
            for kc in s['key_concepts']:
                html += f'<div class="concept-item"><strong>{kc.get("name", "")}</strong>'
                if kc.get('summary'):
                    html += f'<p>{kc["summary"]}</p>'
                if kc.get('why_it_matters'):
                    html += f'<p><em>Why it matters:</em> {kc["why_it_matters"]}</p>'
                html += '</div>'
            html += '</div>'

        if s.get('mental_models'):
            html += '<div class="mental-models"><strong>Mental Models &amp; Frameworks:</strong>'
            for mm in s['mental_models']:
                html += f'<div class="model-item"><strong>{mm.get("name", "")}</strong>'
                if mm.get('how_it_works'):
                    html += f'<p>{mm["how_it_works"]}</p>'
                if mm.get('application'):
                    html += f'<p><em>Application:</em> {mm["application"]}</p>'
                html += '</div>'
            html += '</div>'

        if s.get('quotable_lines'):
            html += '<div class="quotes"><strong>Quotable Lines:</strong>'
            for ql in s['quotable_lines']:
                speaker = f' &mdash; {ql["speaker"]}' if ql.get('speaker') else ''
                html += f'<blockquote>"{ql.get("quote", "")}"<cite>{speaker}</cite>'
                if ql.get('context'):
                    html += f'<p><em>{ql["context"]}</em></p>'
                html += '</blockquote>'
            html += '</div>'

        if s.get('career_relevance'):
            html += '<div class="career"><strong>Career &amp; Work Relevance:</strong><ul>'
            for cr in s['career_relevance']:
                html += f'<li>{cr}</li>'
            html += '</ul></div>'

        verdict = s.get('verdict')
        if verdict:
            novelty = verdict.get('novelty', '—')
            actionability = verdict.get('actionability', '—')
            depth = verdict.get('depth', '—')
            best = verdict.get('best_sections', '')
            html += (
                f'<div class="verdict"><strong>Verdict:</strong> '
                f'Novelty {novelty}/5 &middot; Actionability {actionability}/5 &middot; {depth}'
            )
            if best:
                html += f'<br><em>Best sections to revisit:</em> {best}'
            html += '</div>'

        return html

    def export_email_html(self, summaries: List[Dict[str, Any]], target_date: date) -> str:
        """Export summaries as HTML for email."""
        html = f"""
        <html>
        <head>
            <title>Podcast Digest - {target_date}</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; }}
                h1, h2, h3 {{ color: #333; }}
                .podcast {{ margin-bottom: 2em; }}
                .episode {{ margin-bottom: 1.5em; padding: 1em; background: #f9f9f9; }}
                .one-liner {{ font-style: italic; margin-bottom: 1em; }}
                .concepts, .key-concepts, .mental-models, .quotes, .career, .verdict {{ margin-bottom: 0.75em; }}
                .concept-item, .model-item {{ margin: 0.5em 0 0.5em 1em; }}
                blockquote {{ background: #e8e8e8; padding: 0.5em 1em; margin: 0.5em 0; border-left: 3px solid #aaa; }}
                .verdict {{ background: #eef; padding: 0.5em; border-radius: 4px; }}
                ul {{ margin: 0.5em 0; }}
            </style>
        </head>
        <body>
            <h1>Podcast Digest - {target_date}</h1>
        """

        if not summaries:
            html += "<p>No summaries available for this date.</p>"
        else:
            by_podcast: Dict[str, List] = {}
            for summary in summaries:
                podcast = summary['podcast_title']
                by_podcast.setdefault(podcast, []).append(summary)

            for podcast_name, episodes in by_podcast.items():
                html += f'<div class="podcast"><h2>{podcast_name}</h2>'

                for episode in episodes:
                    html += f'<div class="episode"><h3>{episode["episode_title"]}</h3>'
                    structured = self._parse_structured(episode)
                    if structured:
                        html += self._render_structured_html(structured)
                    else:
                        # Legacy fallback
                        if episode.get('full_summary'):
                            html += f'<div class="one-liner"><strong>Summary:</strong> {episode["full_summary"]}</div>'
                        if episode.get('key_topics'):
                            html += '<div class="concepts"><strong>Key Topics:</strong><ul>'
                            for topic in episode['key_topics']:
                                html += f'<li>{topic}</li>'
                            html += '</ul></div>'
                        if episode.get('quotes'):
                            html += '<div class="quotes"><strong>Notable Quotes:</strong>'
                            for quote in episode['quotes']:
                                html += f'<blockquote>{quote}</blockquote>'
                            html += '</div>'
                    html += '</div>'  # episode

                html += '</div>'  # podcast

        html += """
        </body>
        </html>
        """

        return html
