"""Blog post generation with AP English teacher grading system.

Inspired by Tomasz Tunguz's innovative approach to AI-assisted writing
with iterative grading and improvement loops.
"""

import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from .database import P3Database

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from google import genai as google_genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class BlogWriter:
    def __init__(self, db: P3Database, llm_provider: str = "ollama",
                 llm_model: str = "llama3.2:latest", target_grade: float = 91.0):
        self.db = db
        self.llm_provider = llm_provider.lower()
        self.llm_model = llm_model
        self.target_grade = target_grade
        self.max_iterations = 3

        key_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }
        self.api_key = os.getenv(key_map.get(self.llm_provider, ""))
        
    def generate_blog_post_from_digest(self, topic: str, digest_data: Dict[str, Any], 
                                     context_posts: List[str] = None) -> Dict[str, Any]:
        """Generate blog post from podcast digest with iterative AP English grading.
        
        Args:
            topic: The main topic/angle for the blog post
            digest_data: Structured digest data from podcast analysis
            context_posts: Optional list of related blog posts for style matching
            
        Returns:
            Dict containing final blog post, grades, and iterations
        """
        
        # Extract relevant content from digest
        episode_title = digest_data.get('episode_title', '')
        podcast_title = digest_data.get('podcast_title', '')
        summary = digest_data.get('full_summary', '')
        key_topics = digest_data.get('key_topics', [])
        themes = digest_data.get('themes', [])
        quotes = digest_data.get('quotes', [])
        companies = digest_data.get('startups', [])
        
        # Build context for the blog post
        context = f"""
        Episode: {episode_title} from {podcast_title}
        Summary: {summary}
        Key Topics: {', '.join(key_topics)}
        Themes: {', '.join(themes)}
        Notable Quotes: {quotes}
        Companies Mentioned: {', '.join(companies)}
        """
        
        iterations = []
        current_post = ""
        
        # Generate initial blog post
        initial_prompt = self._build_writing_prompt(topic, context, context_posts)
        current_post = self._generate_with_llm(initial_prompt)
        
        # Iterative grading and improvement (inspired by Tunguz's approach)
        for iteration in range(self.max_iterations):
            grade_result = self._grade_blog_post(current_post)
            iterations.append({
                'iteration': iteration + 1,
                'post': current_post,
                'grade': grade_result['grade'],
                'score': grade_result['score'],
                'feedback': grade_result['feedback']
            })
            
            # Check if we've reached target grade
            if grade_result['score'] >= self.target_grade:
                break
                
            # Improve based on feedback
            if iteration < self.max_iterations - 1:
                improvement_prompt = self._build_improvement_prompt(
                    current_post, grade_result['feedback']
                )
                current_post = self._generate_with_llm(improvement_prompt)
        
        # Generate SEO-friendly slug
        slug = self._generate_slug(topic)
        
        return {
            'final_post': current_post,
            'final_grade': iterations[-1]['grade'],
            'final_score': iterations[-1]['score'],
            'iterations': iterations,
            'topic': topic,
            'slug': slug,
            'metadata': {
                'episode_title': episode_title,
                'podcast_title': podcast_title,
                'generated_at': datetime.now().isoformat(),
                'model_used': self.llm_model
            }
        }
    
    def _build_writing_prompt(self, topic: str, context: str, context_posts: List[str] = None) -> str:
        """Build the initial writing prompt based on Tunguz's style guidelines."""
        
        style_guidelines = """
        Style Guidelines (inspired by Tomasz Tunguz's approach):
        - 500 words or less (49 seconds with reader)
        - No section headers (headers hurt dwell time)
        - Flowing paragraphs that transition smoothly
        - Limit each paragraph to at most two long sentences
        - Strong hook in first few sentences
        - Conclusion that ties back to opening
        - Focus on actionable insights
        - Include specific examples and quotes when relevant
        """
        
        context_section = ""
        if context_posts:
            context_section = f"""
            Related Content for Style Reference:
            {chr(10).join(context_posts[:3])}  # Limit to 3 for context window
            """
        
        return f"""You are an expert blog writer specializing in technology and business content.
        
        {style_guidelines}
        
        Topic: {topic}
        
        Source Material:
        {context}
        
        {context_section}
        
        Write a compelling blog post that:
        1. Opens with a strong hook that draws readers in
        2. Presents insights from the podcast content
        3. Provides actionable takeaways for business/tech readers
        4. Includes relevant quotes to support key points
        5. Concludes with a thought-provoking statement that ties back to the opening
        
        Remember: Be concise, engaging, and focused on delivering value quickly.
        """
    
    def _grade_blog_post(self, blog_post: str) -> Dict[str, Any]:
        """Grade blog post like an AP English teacher (Tunguz's innovation)."""
        
        grading_prompt = f"""You are an experienced AP English teacher grading a blog post. 
        
        Evaluate this blog post and provide:
        1. Letter grade (A+, A, A-, B+, B, B-, C+, C, C-, D+, D, F)
        2. Numerical score (0-100)
        3. Detailed feedback on each criterion
        
        Evaluation Criteria:
        - Hook/Opening (20 points): Does it grab attention immediately?
        - Argument Clarity (20 points): Is the main point clear and well-supported?
        - Evidence and Examples (20 points): Are quotes and examples used effectively?
        - Paragraph Structure (20 points): Do paragraphs flow smoothly with good transitions?
        - Conclusion Strength (20 points): Does it tie back and leave lasting impact?
        - Overall Engagement (bonus/penalty): Would readers stay engaged throughout?
        
        Blog Post to Grade:
        {blog_post}
        
        Format your response as:
        GRADE: [Letter Grade]
        SCORE: [Numerical Score]
        FEEDBACK: [Detailed feedback with specific suggestions for improvement]
        """
        
        response = self._generate_with_llm(grading_prompt)
        
        # Parse response
        grade_match = re.search(r'GRADE:\s*([A-F][+-]?)', response)
        score_match = re.search(r'SCORE:\s*(\d+)', response)
        feedback_match = re.search(r'FEEDBACK:\s*(.*)', response, re.DOTALL)
        
        return {
            'grade': grade_match.group(1) if grade_match else 'C',
            'score': float(score_match.group(1)) if score_match else 75.0,
            'feedback': feedback_match.group(1).strip() if feedback_match else response,
            'raw_response': response
        }
    
    def _build_improvement_prompt(self, current_post: str, feedback: str) -> str:
        """Build prompt to improve blog post based on feedback."""
        
        return f"""You are revising a blog post based on AP English teacher feedback.
        
        Current Blog Post:
        {current_post}
        
        Teacher Feedback:
        {feedback}
        
        Please rewrite the blog post incorporating the feedback while maintaining:
        - The core message and insights
        - Concise, engaging style (500 words or less)
        - Strong hook and conclusion
        - Smooth paragraph transitions
        - Actionable takeaways
        
        Focus especially on addressing the specific issues mentioned in the feedback.
        """
    
    def _generate_slug(self, topic: str) -> str:
        """Generate URL-friendly slug from topic."""
        # Convert to lowercase and replace spaces/special chars with hyphens
        slug = re.sub(r'[^\w\s-]', '', topic.lower())
        slug = re.sub(r'[-\s]+', '-', slug)
        return slug.strip('-')
    
    def _generate_with_llm(self, prompt: str) -> str:
        """Dispatch to the configured LLM provider."""
        if self.llm_provider == "ollama":
            return self._generate_ollama(prompt)
        elif self.llm_provider == "openai":
            return self._generate_openai(prompt)
        elif self.llm_provider == "anthropic":
            return self._generate_anthropic(prompt)
        elif self.llm_provider == "gemini":
            return self._generate_gemini(prompt)
        else:
            return f"Error: unsupported LLM provider '{self.llm_provider}'"

    def _generate_ollama(self, prompt: str) -> str:
        if not OLLAMA_AVAILABLE:
            return "Error: ollama package not installed. Run: pip install ollama"
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert blog writer and writing instructor."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response['message']['content'].strip()
        except Exception as e:
            return f"Error generating content: {e}"

    def _generate_openai(self, prompt: str) -> str:
        if not OPENAI_AVAILABLE:
            return "Error: openai package not installed. Run: pip install openai"
        try:
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert blog writer and writing instructor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating content: {e}"

    def _generate_anthropic(self, prompt: str) -> str:
        if not ANTHROPIC_AVAILABLE:
            return "Error: anthropic package not installed. Run: pip install anthropic"
        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            message = client.messages.create(
                model=self.llm_model,
                max_tokens=2048,
                system="You are an expert blog writer and writing instructor.",
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text.strip()
        except Exception as e:
            return f"Error generating content: {e}"

    def _generate_gemini(self, prompt: str) -> str:
        if not GEMINI_AVAILABLE:
            return "Error: google-genai package not installed. Run: pip install google-genai"
        try:
            client = google_genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=self.llm_model,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            return f"Error generating content: {e}"
    
    def save_blog_post(self, blog_result: Dict[str, Any], output_dir: str = "blog_posts") -> str:
        """Save generated blog post to file."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create filename with date and slug
        date_str = datetime.now().strftime('%Y-%m-%d')
        filename = f"{date_str}-{blog_result['slug']}.md"
        file_path = output_path / filename
        
        # Generate markdown content
        content = f"""---
title: "{blog_result['topic']}"
date: {blog_result['metadata']['generated_at']}
source_episode: "{blog_result['metadata']['episode_title']}"
source_podcast: "{blog_result['metadata']['podcast_title']}"
final_grade: {blog_result['final_grade']}
final_score: {blog_result['final_score']}
model: {blog_result['metadata']['model_used']}
inspired_by: "Tomasz Tunguz's AP English grading system"
---

# {blog_result['topic']}

{blog_result['final_post']}

---

## Generation Notes

- **Final Grade**: {blog_result['final_grade']} ({blog_result['final_score']}/100)
- **Iterations**: {len(blog_result['iterations'])}
- **Source**: {blog_result['metadata']['episode_title']} from {blog_result['metadata']['podcast_title']}
- **Generated**: {blog_result['metadata']['generated_at']}

### Grading History
"""
        
        # Add iteration details
        for iteration in blog_result['iterations']:
            content += f"""
**Iteration {iteration['iteration']}**: {iteration['grade']} ({iteration['score']}/100)
{iteration['feedback'][:200]}...

"""
        
        # Write to file
        with open(file_path, 'w') as f:
            f.write(content)
        
        return str(file_path)

    def generate_social_posts(self, blog_result: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate social media posts from blog content (Tunguz's feature)."""
        
        blog_post = blog_result['final_post']
        topic = blog_result['topic']
        
        # Extract key quotes and insights
        quotes = []
        insights = []
        
        # Simple extraction (could be enhanced with better parsing)
        sentences = blog_post.split('. ')
        for sentence in sentences:
            if len(sentence) > 50 and len(sentence) < 280:  # Twitter length
                if any(word in sentence.lower() for word in ['key', 'important', 'crucial', 'insight']):
                    insights.append(sentence.strip() + '.')
                elif '"' in sentence:
                    quotes.append(sentence.strip())
        
        # Generate Twitter posts
        twitter_prompt = f"""Generate 3 engaging Twitter posts based on this blog post about {topic}.
        
        Blog Post:
        {blog_post}
        
        Requirements:
        - Each post under 280 characters
        - Include relevant hashtags
        - Make them engaging and actionable
        - Reference key insights or quotes when possible
        
        Format as:
        POST 1: [content]
        POST 2: [content]  
        POST 3: [content]
        """
        
        # Generate LinkedIn posts
        linkedin_prompt = f"""Generate 2 LinkedIn posts based on this blog post about {topic}.
        
        Blog Post:
        {blog_post}
        
        Requirements:
        - Professional tone suitable for business audience
        - 100-200 words each
        - Include call-to-action
        - Reference source material appropriately
        
        Format as:
        POST 1: [content]
        POST 2: [content]
        """
        
        twitter_response = self._generate_with_llm(twitter_prompt)
        linkedin_response = self._generate_with_llm(linkedin_prompt)
        
        # Parse responses (simple parsing - could be enhanced)
        twitter_posts = re.findall(r'POST \d+: (.+?)(?=POST \d+:|$)', twitter_response, re.DOTALL)
        linkedin_posts = re.findall(r'POST \d+: (.+?)(?=POST \d+:|$)', linkedin_response, re.DOTALL)
        
        return {
            'twitter': [post.strip() for post in twitter_posts],
            'linkedin': [post.strip() for post in linkedin_posts],
            'quotes': quotes[:3],  # Top 3 quotable excerpts
            'insights': insights[:5]  # Top 5 key insights
        }
