#!/usr/bin/env python3
"""
Generate Lecture Materials from Transcript

This script takes a transcript file and generates:
1. lecture.md - Markdown formatted lecture
2. lecture.ipynb - Jupyter notebook with code cells
3. lecture.html - HTML page for the website

Usage:
    python scripts/generate_lecture.py <module> <lecture_name> [--transcript path/to/transcript.md]

Example:
    python scripts/generate_lecture.py vision how-machines-see-3 --transcript transcript.md

The script expects lectures to be in: lectures/<module>/<lecture_name>/
"""

import argparse
import json
import os
import re
from pathlib import Path
from datetime import datetime

WEBSITE_DIR = Path(__file__).parent.parent
LECTURES_DIR = WEBSITE_DIR / "lectures"


def create_notebook_from_markdown(md_content: str, title: str) -> dict:
    """Convert markdown content to Jupyter notebook format."""
    cells = []

    # Add title cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"# {title}\n"]
    })

    # Split content into sections
    sections = re.split(r'\n(?=##?\s)', md_content)

    for section in sections:
        if not section.strip():
            continue

        # Check if section contains code blocks
        code_blocks = re.findall(r'```python\n(.*?)```', section, re.DOTALL)

        if code_blocks:
            # Split section around code blocks
            parts = re.split(r'```python\n.*?```', section, flags=re.DOTALL)

            for i, part in enumerate(parts):
                if part.strip():
                    cells.append({
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": part.strip().split('\n')
                    })
                if i < len(code_blocks):
                    cells.append({
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": code_blocks[i].strip().split('\n')
                    })
        else:
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": section.strip().split('\n')
            })

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    return notebook


def create_html_from_markdown(md_content: str, title: str, module: str, lecture_name: str) -> str:
    """Create HTML page from markdown content."""

    # Extract sections for summary
    sections = re.findall(r'^##\s+(.+)$', md_content, re.MULTILINE)

    html = f'''---
layout: default
title: "{title}"
---

<nav class="breadcrumb">
  <a href="{{{{ '/' | relative_url }}}}">Home</a> /
  <a href="{{{{ '/lectures.html' | relative_url }}}}">Lectures</a> /
  <a href="{{{{ '/lectures.html#{module}' | relative_url }}}}">{module.title()}</a> /
  {lecture_name}
</nav>

<article class="lecture-content">
  <header>
    <h1>{title}</h1>
    <div class="meta">
      <span>Module: {module.title()}</span>
      <span>Generated: {datetime.now().strftime("%Y-%m-%d")}</span>
    </div>
    <div class="resources">
      <a href="lecture.ipynb" class="btn">Download Notebook</a>
      <a href="lecture.md" class="btn btn-secondary">View Markdown</a>
    </div>
  </header>

  <section class="toc">
    <h2>Contents</h2>
    <ul>
'''

    for section in sections[:10]:  # Limit to 10 sections
        html += f'      <li>{section}</li>\n'

    html += '''    </ul>
  </section>

  <section class="content">
    <p><em>Full content available in the markdown file and Jupyter notebook.</em></p>
  </section>
</article>

<style>
.breadcrumb { font-size: 0.9rem; color: #666; margin-bottom: 1.5rem; }
.breadcrumb a { color: var(--accent-color); }
.lecture-content header { margin-bottom: 2rem; padding-bottom: 1.5rem; border-bottom: 1px solid var(--border-color); }
.meta { font-size: 0.9rem; color: #888; margin-bottom: 1rem; }
.meta span { margin-right: 1.5rem; }
.btn-secondary { background: #e9ecef; color: var(--primary-color); }
.toc { background: var(--light-bg); padding: 1.5rem; border-radius: 6px; }
.toc h2 { margin-top: 0; }
</style>
'''
    return html


def process_transcript(transcript_path: str) -> str:
    """Process a raw transcript into formatted markdown.

    This is a placeholder - in production, this would call an LLM
    to transform the transcript into well-structured lecture content.
    """
    with open(transcript_path, 'r') as f:
        content = f.read()

    # For now, just return the content as-is
    # In production: call Claude API to format the transcript
    print("Note: Transcript processing with LLM not yet implemented.")
    print("Using transcript content directly.")

    return content


def update_lectures_page(module: str, lecture_name: str, title: str, description: str):
    """Update lectures.html with new lecture entry.

    This adds a new lecture card to the appropriate module section.
    """
    lectures_html = WEBSITE_DIR / "lectures.html"

    with open(lectures_html, 'r') as f:
        content = f.read()

    # Create new lecture card HTML
    new_card = f'''
    <div class="lecture-card">
      <h3><a href="{{{{ '/lectures/{module}/{lecture_name}/lecture.html' | relative_url }}}}">{ title}</a></h3>
      <p class="description">{description}</p>
      <div class="resources">
        <a href="{{{{ '/lectures/{module}/{lecture_name}/lecture.ipynb' | relative_url }}}}" class="btn-small">Notebook</a>
        <a href="{{{{ '/lectures/{module}/{lecture_name}/lecture.md' | relative_url }}}}" class="btn-small">Markdown</a>
      </div>
    </div>
'''

    # Find the module section and add the card
    # This is a simple implementation - could be improved
    section_marker = f'id="{module}"'
    if section_marker in content:
        # Find the end of the lecture-list div
        pattern = f'(id="{module}".*?<div class="lecture-list">)(.*?)(</div>\\s*</section>)'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            # Insert new card before closing div
            new_content = content[:match.end(2)] + new_card + content[match.start(3):]
            with open(lectures_html, 'w') as f:
                f.write(new_content)
            print(f"Updated lectures.html with new entry for {lecture_name}")
            return

    print(f"Warning: Could not find section '{module}' in lectures.html")
    print("Please add the lecture entry manually.")


def main():
    parser = argparse.ArgumentParser(description="Generate lecture materials from transcript")
    parser.add_argument("module", help="Module name (e.g., vision, language, timeseries)")
    parser.add_argument("lecture_name", help="Lecture folder name (e.g., how-machines-see-3)")
    parser.add_argument("--transcript", "-t", help="Path to transcript file")
    parser.add_argument("--title", help="Lecture title")
    parser.add_argument("--description", help="Short description for lectures page")

    args = parser.parse_args()

    # Create lecture directory
    lecture_dir = LECTURES_DIR / args.module / args.lecture_name
    lecture_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating lecture in: {lecture_dir}")

    # Process transcript if provided
    if args.transcript:
        transcript_path = Path(args.transcript)
        if not transcript_path.exists():
            # Check if it's relative to lecture_dir
            transcript_path = lecture_dir / args.transcript
        if transcript_path.exists():
            md_content = process_transcript(str(transcript_path))
            # Copy transcript
            (lecture_dir / "transcript.md").write_text(md_content)
        else:
            print(f"Warning: Transcript file not found: {args.transcript}")
            md_content = f"# {args.title or args.lecture_name}\n\n[Transcript content here]"
    else:
        md_content = f"# {args.title or args.lecture_name}\n\n[Lecture content here]"

    title = args.title or args.lecture_name.replace("-", " ").title()
    description = args.description or "Lecture description"

    # Generate markdown file
    md_path = lecture_dir / "lecture.md"
    md_path.write_text(md_content)
    print(f"Created: {md_path}")

    # Generate Jupyter notebook
    notebook = create_notebook_from_markdown(md_content, title)
    ipynb_path = lecture_dir / "lecture.ipynb"
    with open(ipynb_path, 'w') as f:
        json.dump(notebook, f, indent=2)
    print(f"Created: {ipynb_path}")

    # Generate HTML page
    html_content = create_html_from_markdown(md_content, title, args.module, args.lecture_name)
    html_path = lecture_dir / "lecture.html"
    html_path.write_text(html_content)
    print(f"Created: {html_path}")

    # Update lectures.html
    # update_lectures_page(args.module, args.lecture_name, title, description)

    print(f"\nLecture generated successfully!")
    print(f"Location: {lecture_dir}")
    print(f"\nFiles created:")
    print(f"  - lecture.md")
    print(f"  - lecture.ipynb")
    print(f"  - lecture.html")
    print(f"\nNext steps:")
    print(f"  1. Edit lecture.md with full content")
    print(f"  2. Re-run this script to regenerate notebook and HTML")
    print(f"  3. Add entry to lectures.html if not auto-added")


if __name__ == "__main__":
    main()
