"""Command-line interface for P³."""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import click
import yaml
from rich.console import Console
from rich.table import Table
from rich.progress import track

from .database import P3Database
from .downloader import PodcastDownloader
from .transcriber import AudioTranscriber
from .cleaner import TranscriptCleaner
from .exporter import DigestExporter
from .writer import BlogWriter

console = Console()


def load_config(config_path: str = "config/feeds.yaml"):
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        console.print("Copy config/feeds.yaml.example to config/feeds.yaml and configure your feeds")
        sys.exit(1)
    
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        sys.exit(1)


@click.group()
@click.option('--config', default="config/feeds.yaml", help='Configuration file path')
@click.option('--db', default="data/p3.duckdb", help='Database file path')
@click.pass_context
def main(ctx, config, db):
    """Parakeet Podcast Processor (P³) - Automated podcast processing."""
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['db_path'] = db
    ctx.obj['db'] = P3Database(db)


@main.command()
@click.option('--max-episodes', default=None, type=int, help='Max episodes per feed')
@click.pass_context
def fetch(ctx, max_episodes):
    """Download new podcast episodes from configured RSS feeds."""
    config = load_config(ctx.obj['config_path'])
    db = ctx.obj['db']
    
    settings = config.get('settings', {})
    max_eps = max_episodes or settings.get('max_episodes_per_feed', 10)
    
    downloader = PodcastDownloader(
        db=db,
        max_episodes=max_eps,
        audio_format=settings.get('audio_format', 'wav')
    )
    
    console.print("[blue]Fetching podcast episodes...[/blue]")
    
    feeds = config.get('feeds', [])
    if not feeds:
        console.print("[yellow]No feeds configured[/yellow]")
        return
    
    total_downloaded = 0
    results = downloader.fetch_all_feeds(feeds)
    
    # Display results table
    table = Table(title="Download Results")
    table.add_column("Podcast", style="cyan")
    table.add_column("New Episodes", style="green", justify="right")
    
    for name, count in results.items():
        table.add_row(name, str(count))
        total_downloaded += count
    
    console.print(table)
    console.print(f"[green]Total downloaded: {total_downloaded} episodes[/green]")


@main.command()
@click.option('--model', default=None, help='Whisper model to use (local-whisper only)')
@click.option('--episode-id', type=int, help='Transcribe specific episode')
@click.pass_context
def transcribe(ctx, model, episode_id):
    """Transcribe downloaded audio files."""
    config = load_config(ctx.obj['config_path'])
    db = ctx.obj['db']

    settings = config.get('settings', {})
    whisper_model = model or settings.get('whisper_model', 'base')

    # Support both old (parakeet_enabled) and new (transcription_provider) config keys
    transcription_provider = settings.get('transcription_provider')
    if transcription_provider is None:
        use_parakeet = settings.get('parakeet_enabled', False)
        transcription_provider = 'local-parakeet' if use_parakeet else 'local-whisper'

    provider_labels = {
        'local-parakeet': 'Parakeet MLX (local)',
        'local-whisper':  f'Whisper {whisper_model} (local)',
        'openai-api':     'OpenAI Whisper API (cloud)',
    }
    console.print(f"[blue]Transcription provider: {provider_labels.get(transcription_provider, transcription_provider)}[/blue]")

    cleanup_audio = settings.get('cleanup_old_files', False)

    # Build a lookup: podcast name → save_transcript flag
    transcript_feeds = {
        f['name']: f.get('save_transcript', False)
        for f in config.get('feeds', [])
    }

    transcriber = AudioTranscriber(
        db=db,
        whisper_model=whisper_model,
        use_parakeet=(transcription_provider == 'local-parakeet'),
        parakeet_model=settings.get('parakeet_model', 'mlx-community/parakeet-tdt-0.6b-v2'),
        transcription_provider=transcription_provider,
        cleanup_audio=cleanup_audio,
        parakeet_chunk_duration=settings.get('parakeet_chunk_duration', 600),
    )

    def _maybe_save_transcript(ep_id):
        episode = db.get_episode_by_id(ep_id)
        if episode and transcript_feeds.get(episode.get('podcast_title', ''), False):
            from .exporter import write_transcript
            path = write_transcript(db, ep_id)
            if path:
                console.print(f"[cyan]  Transcript saved: {path}[/cyan]")

    if episode_id:
        console.print(f"[blue]Transcribing episode {episode_id}...[/blue]")
        success = transcriber.transcribe_episode(episode_id)
        if success:
            console.print(f"[green]✓ Episode {episode_id} transcribed[/green]")
            _maybe_save_transcript(episode_id)
        else:
            console.print(f"[red]✗ Failed to transcribe episode {episode_id}[/red]")
    else:
        console.print("[blue]Transcribing all pending episodes...[/blue]")
        episodes = db.get_episodes_by_status('downloaded')

        if not episodes:
            console.print("[yellow]No episodes to transcribe[/yellow]")
            return

        transcribed = 0
        for episode in track(episodes, description="Transcribing..."):
            if transcriber.transcribe_episode(episode['id']):
                transcribed += 1
                _maybe_save_transcript(episode['id'])

        console.print(f"[green]Transcribed {transcribed} episodes[/green]")


@main.command()
@click.option('--provider', default=None, help='LLM provider (openai, anthropic, ollama)')
@click.option('--model', default=None, help='LLM model to use')
@click.option('--episode-id', type=int, help='Process specific episode')
@click.pass_context
def digest(ctx, provider, model, episode_id):
    """Generate structured summaries from transcripts."""
    config = load_config(ctx.obj['config_path'])
    db = ctx.obj['db']
    
    settings = config.get('settings', {})
    llm_provider = provider or settings.get('llm_provider', 'openai')
    llm_model = model or settings.get('llm_model', 'gpt-3.5-turbo')
    
    cleaner = TranscriptCleaner(
        db=db,
        llm_provider=llm_provider,
        llm_model=llm_model,
        ollama_base_url=settings.get('ollama_base_url', 'http://localhost:11434')
    )
    
    if episode_id:
        console.print(f"[blue]Processing episode {episode_id}...[/blue]")
        result = cleaner.generate_summary(episode_id)
        if result:
            console.print(f"[green]✓ Episode {episode_id} processed[/green]")
        else:
            console.print(f"[red]✗ Failed to process episode {episode_id}[/red]")
    else:
        console.print("[blue]Processing all transcribed episodes...[/blue]")
        processed = cleaner.process_all_transcribed()
        console.print(f"[green]Processed {processed} episodes[/green]")


@main.command()
@click.option('--date', help='Export date (YYYY-MM-DD)')
@click.option('--format', multiple=True, help='Export format (markdown, json)')
@click.option('--output', help='Output file path')
@click.pass_context
def export(ctx, date, format, output):
    """Export daily digest summaries."""
    config = load_config(ctx.obj['config_path'])
    db = ctx.obj['db']
    
    # Parse date
    if date:
        try:
            target_date = datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            console.print("[red]Invalid date format. Use YYYY-MM-DD[/red]")
            return
    else:
        target_date = datetime.now()
    
    # Get export formats
    formats = list(format) if format else config.get('settings', {}).get('export_format', ['markdown'])
    
    exporter = DigestExporter(db)
    
    summaries = db.get_summaries_by_date(target_date)
    
    if not summaries:
        console.print(f"[yellow]No summaries found for {target_date.date()}[/yellow]")
        return
    
    console.print(f"[blue]Exporting {len(summaries)} summaries for {target_date.date()}[/blue]")

    # Build a filename-safe podcast label from unique podcast names in the export
    def _podcast_label(summaries_list):
        import re
        names = list(dict.fromkeys(s['podcast_title'] for s in summaries_list))
        slug = "_".join(
            re.sub(r'[^A-Za-z0-9]+', '-', name).strip('-')
            for name in names[:3]  # cap at 3 names to keep filenames sane
        )
        return slug or "digest"

    for fmt in formats:
        date_str = target_date.strftime('%Y-%m-%d')
        podcast_slug = _podcast_label(summaries)
        if fmt == 'markdown':
            if output:
                # If an explicit output path is given, write a single combined file
                content = exporter.export_markdown(summaries, target_date.date())
                with open(output, 'w') as f:
                    f.write(content)
                console.print(f"[green]✓ Exported markdown: {output}[/green]")
            else:
                # Default: one file per episode
                for episode in summaries:
                    filename = exporter.episode_filename(episode, target_date.date())
                    content = exporter.export_episode_markdown(episode, target_date.date())
                    with open(filename, 'w') as f:
                        f.write(content)
                    console.print(f"[green]✓ Exported markdown: {filename}[/green]")
        elif fmt == 'json':
            content = exporter.export_json(summaries, target_date.date())
            filename = output or f"{date_str}_{podcast_slug}.json"
            with open(filename, 'w') as f:
                f.write(content)
            console.print(f"[green]✓ Exported json: {filename}[/green]")
        else:
            console.print(f"[red]Unsupported format: {fmt}[/red]")
            continue


@main.command()
@click.option('--episode-id', type=int, default=None,
              help='Episode ID to export (from p3 status --detail)')
@click.option('--output-dir', default='transcripts', show_default=True,
              help='Directory to write transcript files')
@click.option('--all-processed', is_flag=True, default=False,
              help='Export transcripts for every processed episode')
@click.pass_context
def transcript(ctx, episode_id, output_dir, all_processed):
    """Export full transcripts to readable markdown files.

    Each file is named YYYY-MM-DD_PodcastName_EpisodeTitle_transcript.md
    and contains every transcribed segment with [HH:MM:SS] timestamps.

    Use --episode-id for a specific episode, or --all-processed to export
    every episode in the database regardless of the save_transcript setting.
    """
    from .exporter import write_transcript
    db = ctx.obj['db']

    if all_processed:
        episodes = db.get_episodes_by_status('processed')
        if not episodes:
            console.print("[yellow]No processed episodes found[/yellow]")
            return
        saved = 0
        for ep in episodes:
            path = write_transcript(db, ep['id'], output_dir=output_dir)
            if path:
                console.print(f"[green]✓[/green] {path}")
                saved += 1
        console.print(f"\n[green]{saved} transcript(s) written to {output_dir}/[/green]")
        return

    if not episode_id:
        console.print("[red]Provide --episode-id N or --all-processed[/red]")
        console.print("Tip: run 'p3 status --detail' to see episode IDs")
        return

    path = write_transcript(db, episode_id, output_dir=output_dir)
    if path:
        console.print(f"[green]✓ Transcript saved: {path}[/green]")
    else:
        console.print(f"[red]No transcript found for episode {episode_id}[/red]")
        console.print("Tip: run 'p3 status --detail' to confirm the episode exists")


@main.command()
@click.option('--detail', is_flag=True, default=False, help='Show individual episodes, not just counts')
@click.option('--filter', 'status_filter', default=None,
              type=click.Choice(['downloaded', 'transcribed', 'processed']),
              help='Show only episodes with this status')
@click.pass_context
def status(ctx, detail, status_filter):
    """Show processing status of episodes.

    By default shows a summary count table. Use --detail to list every
    episode with its podcast, date, and current stage. Use --filter to
    narrow to a specific stage (e.g. --filter transcribed).
    """
    db = ctx.obj['db']

    statuses = ['downloaded', 'transcribed', 'processed', 'failed']
    all_episodes: dict = {}
    for s in statuses:
        all_episodes[s] = db.get_episodes_by_status(s)

    failed_count = len(all_episodes['failed'])

    # ── Summary count table (always shown) ────────────────────────────────────
    status_styles = {
        'downloaded':  'yellow',
        'transcribed': 'blue',
        'processed':   'green',
        'failed':      'red',
    }
    meta = {
        'downloaded':  "Audio saved — waiting for transcription",
        'transcribed': "Transcript ready — waiting for LLM digest",
        'processed':   "Summary generated — export-ready",
        'failed':      "Error recorded — run 'p3 errors' for details",
    }
    total = sum(len(v) for v in all_episodes.values())
    summary = Table(title="Episode Pipeline Status", show_footer=True)
    summary.add_column("Stage", style="bold")
    summary.add_column("Count", justify="right", footer=str(total))
    summary.add_column("Meaning")
    for s in statuses:
        count = len(all_episodes[s])
        if count == 0 and s == 'failed':
            continue  # hide the failed row when nothing has failed
        summary.add_row(
            f"[{status_styles[s]}]{s.title()}[/{status_styles[s]}]",
            f"[{status_styles[s]}]{count}[/{status_styles[s]}]",
            meta[s],
        )
    console.print(summary)

    if failed_count:
        console.print(f"[bold red]⚠  {failed_count} episode(s) failed — run 'p3 errors' to see details[/bold red]")

    # ── Detail table ──────────────────────────────────────────────────────────
    if detail or status_filter:
        stages = [status_filter] if status_filter else statuses
        for s in stages:
            episodes = all_episodes[s]
            if not episodes:
                continue
            detail_table = Table(
                title=f"[{status_styles[s]}]{s.title()}[/{status_styles[s]}] — {len(episodes)} episode(s)",
                show_lines=True,
            )
            detail_table.add_column("ID", style="dim", justify="right", no_wrap=True)
            detail_table.add_column("Podcast", style="cyan", no_wrap=True)
            detail_table.add_column("Episode Title")
            detail_table.add_column("Date", no_wrap=True)
            detail_table.add_column("File", style="dim")

            for ep in episodes:
                ep_date = str(ep.get('date', ''))[:10] if ep.get('date') else '—'
                file_path = ep.get('file_path') or '—'
                if len(file_path) > 40:
                    file_path = '…' + file_path[-38:]
                detail_table.add_row(
                    str(ep['id']),
                    ep.get('podcast_title', '—'),
                    ep.get('title', '—'),
                    ep_date,
                    file_path,
                )
            console.print(detail_table)


@main.command()
@click.option('--episode-id', type=int, default=None, help='Show errors for a specific episode')
@click.option('--retry', is_flag=True, default=False, help='Re-queue failed episodes back to downloaded so they can be retried')
@click.pass_context
def errors(ctx, episode_id, retry):
    """Show the error log for failed episodes.

    Each entry records which stage failed (transcription or digest),
    the exception type, and the full error message.

    Use --retry to reset failed episodes back to 'downloaded' so they
    will be picked up again on the next 'p3 transcribe' or 'p3 run'.
    """
    db = ctx.obj['db']

    if retry:
        failed = db.get_failed_episodes()
        if not failed:
            console.print("[green]No failed episodes to retry.[/green]")
            return
        for ep in failed:
            db.update_episode_status(ep['id'], 'downloaded')
            console.print(f"[yellow]Re-queued:[/yellow] {ep['title']} (id={ep['id']})")
        console.print(f"[green]✓ {len(failed)} episode(s) reset to 'downloaded' — run 'p3 transcribe' to retry[/green]")
        return

    error_list = db.get_errors(episode_id=episode_id)

    if not error_list:
        msg = f"for episode {episode_id}" if episode_id else "— all clear"
        console.print(f"[green]No errors recorded {msg}[/green]")
        return

    table = Table(
        title=f"[red]Error Log — {len(error_list)} record(s)[/red]",
        show_lines=True,
    )
    table.add_column("ID", style="dim", justify="right", no_wrap=True)
    table.add_column("Ep ID", justify="right", no_wrap=True)
    table.add_column("Podcast", style="cyan", no_wrap=True)
    table.add_column("Episode Title")
    table.add_column("Stage", no_wrap=True)
    table.add_column("Error Type", style="red", no_wrap=True)
    table.add_column("Message")
    table.add_column("When", no_wrap=True)

    for err in error_list:
        when = str(err.get('created_at', ''))[:16]
        msg = err['error_message']
        if len(msg) > 120:
            msg = msg[:117] + '…'
        table.add_row(
            str(err['id']),
            str(err['episode_id']),
            err['podcast_title'],
            err['episode_title'],
            err['stage'],
            err['error_type'],
            msg,
            when,
        )

    console.print(table)
    console.print("\n[dim]Tip: use 'p3 errors --retry' to re-queue all failed episodes[/dim]")


@main.command()
@click.option('--topic', required=True, help='Blog post topic/angle')
@click.option('--date', help='Date to use for digest (YYYY-MM-DD), defaults to today')
@click.option('--target-grade', default=91.0, help='Target grade for AP English teacher (default: 91.0)')
@click.pass_context
def write(ctx, topic, date, target_grade):
    """Generate blog post from podcast digest using AP English grading system.
    
    Inspired by Tomasz Tunguz's innovative iterative writing approach.
    """
    config = load_config(ctx.obj['config_path'])
    db = ctx.obj['db']
    
    settings = config.get('settings', {})
    llm_provider = settings.get('llm_provider', 'ollama')
    llm_model = settings.get('llm_model', 'llama3.2:latest')
    
    # Parse date
    if date:
        try:
            target_date = datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            console.print("[red]Invalid date format. Use YYYY-MM-DD[/red]")
            return
    else:
        target_date = datetime.now()
    
    # Get summaries for the date
    summaries = db.get_summaries_by_date(target_date)
    
    if not summaries:
        console.print(f"[yellow]No summaries found for {target_date.date()}[/yellow]")
        console.print("Run 'p3 digest' first to generate summaries")
        return
    
    writer = BlogWriter(
        db=db,
        llm_provider=llm_provider,
        llm_model=llm_model,
        target_grade=target_grade
    )
    
    console.print(f"[blue]Generating blog post: '{topic}'[/blue]")
    console.print(f"Using {len(summaries)} podcast summaries from {target_date.date()}")
    console.print(f"Target grade: {target_grade}/100 (inspired by Tomasz Tunguz)")
    
    # Use the first summary as primary source (could be enhanced to combine multiple)
    primary_summary = summaries[0]
    
    with console.status("[bold green]Writing and grading blog post..."):
        blog_result = writer.generate_blog_post_from_digest(topic, primary_summary)
    
    # Show results
    console.print(f"\n[green]✓ Blog post generated![/green]")
    console.print(f"Final Grade: {blog_result['final_grade']} ({blog_result['final_score']}/100)")
    console.print(f"Iterations: {len(blog_result['iterations'])}")
    
    # Save blog post
    file_path = writer.save_blog_post(blog_result)
    console.print(f"Saved to: {file_path}")
    
    # Generate social media posts
    console.print(f"\n[blue]Generating social media posts...[/blue]")
    social_posts = writer.generate_social_posts(blog_result)
    
    # Display social posts
    console.print("\n[cyan]📱 Twitter Posts:[/cyan]")
    for i, post in enumerate(social_posts['twitter'], 1):
        console.print(f"{i}. {post}")
    
    console.print("\n[cyan]💼 LinkedIn Posts:[/cyan]")
    for i, post in enumerate(social_posts['linkedin'], 1):
        console.print(f"{i}. {post[:100]}...")
    
    # Show final blog post preview
    console.print(f"\n[cyan]📄 Blog Post Preview:[/cyan]")
    console.print("-" * 50)
    preview = blog_result['final_post'][:500]
    console.print(f"{preview}...")
    console.print("-" * 50)
    console.print(f"[green]Complete post saved to: {file_path}[/green]")


@main.command()
@click.pass_context
def init(ctx):
    """Initialize P³ — interactive setup for transcription and LLM provider."""
    console.print("[bold blue]Parakeet Podcast Processor (P³) — Setup[/bold blue]\n")

    # ── Directories ──────────────────────────────────────────────────────────
    dirs = ['data', 'config', 'logs', 'data/audio', 'exports', 'blog_posts']
    for dir_name in dirs:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
    console.print("✓ Directories created\n")

    # ── Transcription provider ────────────────────────────────────────────────
    console.print("[bold cyan]Step 1 — Transcription  (audio → text)[/bold cyan]")
    console.print("  [green]local-parakeet[/green]  Apple Silicon only · fastest · fully offline")
    console.print("  [green]local-whisper[/green]   Any machine · fully offline · slower")
    console.print("  [green]openai-api[/green]      Cloud · ~$0.006/min · no GPU needed\n")

    transcription_provider = click.prompt(
        "Transcription provider",
        type=click.Choice(['local-parakeet', 'local-whisper', 'openai-api'], case_sensitive=False),
        default='local-parakeet',
    )

    whisper_model = 'base'
    parakeet_enabled = False
    parakeet_model = 'mlx-community/parakeet-tdt-0.6b-v2'

    if transcription_provider == 'local-parakeet':
        parakeet_enabled = True
        parakeet_model = click.prompt(
            "  Parakeet model", default='mlx-community/parakeet-tdt-0.6b-v2'
        )
    elif transcription_provider == 'local-whisper':
        whisper_model = click.prompt(
            "  Whisper model size",
            type=click.Choice(['tiny', 'base', 'small', 'medium', 'large'], case_sensitive=False),
            default='base',
        )
    else:
        openai_key = os.environ.get('OPENAI_API_KEY')
        if openai_key:
            console.print("  [green]✓ OPENAI_API_KEY found[/green]")
        else:
            console.print("  [yellow]⚠  Set OPENAI_API_KEY before running 'p3 transcribe'[/yellow]")

    # ── LLM provider ─────────────────────────────────────────────────────────
    console.print("\n[bold cyan]Step 2 — LLM Provider  (transcript → summary + blog)[/bold cyan]")
    console.print("  [green]ollama[/green]     Fully local · free · requires Ollama running")
    console.print("  [green]openai[/green]     GPT-4o-mini / GPT-4o · ~$0.15–5 / 1M tokens")
    console.print("  [green]anthropic[/green]  Claude Haiku / Sonnet · ~$0.25–15 / 1M tokens")
    console.print("  [green]gemini[/green]     Gemini Flash / Pro · ~$0.075–7 / 1M tokens\n")

    llm_provider = click.prompt(
        "LLM provider",
        type=click.Choice(['ollama', 'openai', 'anthropic', 'gemini'], case_sensitive=False),
        default='ollama',
    )

    model_defaults = {
        'ollama':    'gemma3:12b',
        'openai':    'gpt-4o-mini',
        'anthropic': 'claude-3-5-haiku-20241022',
        'gemini':    'gemini-2.0-flash',
    }
    llm_model = click.prompt("  Model name", default=model_defaults[llm_provider])

    api_key_envs = {
        'openai':    'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'gemini':    'GEMINI_API_KEY',
    }
    if llm_provider in api_key_envs:
        env_var = api_key_envs[llm_provider]
        if os.environ.get(env_var):
            console.print(f"  [green]✓ {env_var} found[/green]")
        else:
            console.print(f"  [yellow]⚠  Set {env_var} before running 'p3 digest'[/yellow]")
    else:
        console.print(f"  [yellow]⚠  Ensure Ollama is running: ollama serve[/yellow]")
        console.print(f"  [yellow]⚠  Pull the model first: ollama pull {llm_model}[/yellow]")

    # ── Write config/feeds.yaml ───────────────────────────────────────────────
    console.print()
    config_path = Path("config/feeds.yaml")

    config_data = {
        'feeds': [
            {
                'name': 'Example Podcast',
                'url': 'https://example.com/feed/podcast',
                'category': 'tech',
            }
        ],
        'settings': {
            'audio_format': 'wav',
            'max_episodes_per_feed': 1,
            'cleanup_old_files': True,
            'cleanup_days': 30,
            'transcription_provider': transcription_provider,
            'whisper_model': whisper_model,
            'parakeet_enabled': parakeet_enabled,
            'parakeet_model': parakeet_model,
            'llm_provider': llm_provider,
            'llm_model': llm_model,
            'ollama_base_url': 'http://localhost:11434',
            'export_format': ['markdown', 'json'],
            'email_enabled': False,
        },
    }

    if not config_path.exists() or click.confirm(
        "config/feeds.yaml already exists — overwrite?", default=False
    ):
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        console.print("✓ Wrote config/feeds.yaml")
    else:
        console.print("[yellow]Skipped config/feeds.yaml (kept existing)[/yellow]")

    # ── Database ──────────────────────────────────────────────────────────────
    db = P3Database("data/p3.duckdb")
    db.close()
    console.print("✓ Initialized database\n")

    console.print("[bold green]P³ is ready![/bold green]")
    console.print("\nNext steps:")
    console.print("  1. Edit [cyan]config/feeds.yaml[/cyan] — replace the example feed URL with your RSS feeds")
    console.print("  2. [cyan]p3 fetch[/cyan]                   — download episodes")
    console.print("  3. [cyan]p3 transcribe[/cyan]              — transcribe audio")
    console.print("  4. [cyan]p3 digest[/cyan]                  — generate summaries")
    console.print("  5. [cyan]p3 write --topic \"...\"[/cyan]     — generate blog post")
    console.print("  Or run everything at once: [cyan]p3 run[/cyan]")


@main.command()
@click.option('--max-episodes', default=None, type=int, help='Max episodes per feed (overrides config)')
@click.option('--output-dir', default='.', show_default=True, help='Directory to write per-episode markdown files')
@click.pass_context
def run(ctx, max_episodes, output_dir):
    """Run the full pipeline: fetch → transcribe → digest → export.

    Writes one markdown file per episode into OUTPUT_DIR, named:
    YYYY-MM-DD_PodcastName_EpisodeTitle.md
    """
    import os
    from pathlib import Path

    config = load_config(ctx.obj['config_path'])
    db = ctx.obj['db']
    settings = config.get('settings', {})

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Fetch ───────────────────────────────────────────────────────────────
    console.rule("[bold blue]Step 1 — Fetch[/bold blue]")
    max_eps = max_episodes or settings.get('max_episodes_per_feed', 10)
    downloader = PodcastDownloader(
        db=db,
        max_episodes=max_eps,
        audio_format=settings.get('audio_format', 'wav'),
    )
    feeds = config.get('feeds', [])
    if not feeds:
        console.print("[yellow]No feeds configured — edit config/feeds.yaml[/yellow]")
        return

    total_downloaded = 0
    results = downloader.fetch_all_feeds(feeds)
    table = Table()
    table.add_column("Podcast", style="cyan")
    table.add_column("New Episodes", style="green", justify="right")
    for name, count in results.items():
        table.add_row(name, str(count))
        total_downloaded += count
    console.print(table)
    console.print(f"[green]Downloaded {total_downloaded} new episodes[/green]")

    # ── 2. Transcribe ──────────────────────────────────────────────────────────
    console.rule("[bold blue]Step 2 — Transcribe[/bold blue]")
    transcription_provider = settings.get('transcription_provider', 'local-whisper')
    whisper_model = settings.get('whisper_model', 'base')
    provider_labels = {
        'local-parakeet': 'Parakeet MLX (local)',
        'local-whisper': f'Whisper {whisper_model} (local)',
        'openai-api': 'OpenAI Whisper API (cloud)',
    }
    console.print(f"Provider: {provider_labels.get(transcription_provider, transcription_provider)}")

    transcriber = AudioTranscriber(
        db=db,
        whisper_model=whisper_model,
        use_parakeet=(transcription_provider == 'local-parakeet'),
        parakeet_model=settings.get('parakeet_model', 'mlx-community/parakeet-tdt-0.6b-v2'),
        transcription_provider=transcription_provider,
        cleanup_audio=settings.get('cleanup_old_files', False),
        parakeet_chunk_duration=settings.get('parakeet_chunk_duration', 600),
    )
    episodes_to_transcribe = db.get_episodes_by_status('downloaded')
    # Build transcript save lookup for the run command too
    run_transcript_feeds = {
        f['name']: f.get('save_transcript', False)
        for f in feeds
    }

    if not episodes_to_transcribe:
        console.print("[yellow]No episodes to transcribe[/yellow]")
    else:
        transcribed = 0
        for episode in track(episodes_to_transcribe, description="Transcribing..."):
            if transcriber.transcribe_episode(episode['id']):
                transcribed += 1
                if run_transcript_feeds.get(episode.get('podcast_title', ''), False):
                    from .exporter import write_transcript
                    path = write_transcript(db, episode['id'])
                    if path:
                        console.print(f"[cyan]  Transcript saved: {path}[/cyan]")
        console.print(f"[green]Transcribed {transcribed} episodes[/green]")

    # ── 3. Digest ──────────────────────────────────────────────────────────────
    console.rule("[bold blue]Step 3 — Digest[/bold blue]")
    llm_provider = settings.get('llm_provider', 'openai')
    llm_model = settings.get('llm_model', 'gpt-3.5-turbo')
    console.print(f"LLM: {llm_provider} / {llm_model}")

    cleaner = TranscriptCleaner(
        db=db,
        llm_provider=llm_provider,
        llm_model=llm_model,
        ollama_base_url=settings.get('ollama_base_url', 'http://localhost:11434'),
    )
    processed = cleaner.process_all_transcribed()
    console.print(f"[green]Summarized {processed} episodes[/green]")

    # ── 4. Export ──────────────────────────────────────────────────────────────
    console.rule("[bold blue]Step 4 — Export[/bold blue]")
    target_date = datetime.now()
    summaries = db.get_summaries_by_date(target_date)

    if not summaries:
        console.print("[yellow]No summaries for today — nothing to export[/yellow]")
        return

    exporter = DigestExporter(db)
    exported = 0
    for episode in summaries:
        filename = out_dir / exporter.episode_filename(episode, target_date.date())
        content = exporter.export_episode_markdown(episode, target_date.date())
        filename.write_text(content)
        console.print(f"[green]✓ {filename}[/green]")
        exported += 1

    console.rule()
    console.print(
        f"[bold green]Done.[/bold green] "
        f"{total_downloaded} fetched · {processed} summarized · {exported} files written to [cyan]{out_dir}[/cyan]"
    )


if __name__ == '__main__':
    main()
