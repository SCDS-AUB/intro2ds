#!/usr/bin/env python3
"""
SCDS Lecture Sync System

Converts Jupyter notebooks to Jekyll-compatible markdown with:
- Proper frontmatter for Jekyll
- LaTeX math delimiter handling
- Image extraction and path fixing
- Code syntax highlighting

Structure expected:
    lectures/<module>/<lecture-name>/
        lecture.ipynb    # Source notebook
        transcript.md    # Optional transcript
        → lecture.md     # Generated output

Usage:
    python scripts/sync_lectures.py [--dry-run] [--verbose]
"""

import os
import re
import json
import base64
import argparse
import logging
from pathlib import Path
from typing import Optional

# Notebook conversion
try:
    from nbconvert import MarkdownExporter
    import nbformat
    NOTEBOOK_SUPPORT = True
except ImportError:
    NOTEBOOK_SUPPORT = False
    print("Warning: nbconvert not installed. Run: pip install nbconvert nbformat")

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class LectureSync:
    def __init__(self, website_dir: Path, dry_run: bool = False):
        self.website_dir = Path(website_dir)
        self.lectures_dir = self.website_dir / "lectures"
        self.dry_run = dry_run

        # Load config for GitHub settings
        self.github_repo = None
        self.github_branch = 'main'
        config_path = self.website_dir / "_config.yml"
        if config_path.exists():
            try:
                import yaml
                with open(config_path) as f:
                    config = yaml.safe_load(f) or {}
                self.github_repo = config.get('github_repo')
                self.github_branch = config.get('github_branch', 'main')
            except:
                pass

    def find_lectures(self):
        """Find all lecture notebooks in the lectures directory."""
        lectures = []
        for module_dir in self.lectures_dir.iterdir():
            if not module_dir.is_dir() or module_dir.name.startswith('.'):
                continue
            for lecture_dir in module_dir.iterdir():
                if not lecture_dir.is_dir():
                    continue
                ipynb = lecture_dir / "lecture.ipynb"
                if ipynb.exists():
                    lectures.append({
                        'module': module_dir.name,
                        'lecture': lecture_dir.name,
                        'path': lecture_dir,
                        'notebook': ipynb
                    })
        return lectures

    def extract_title(self, content: str, fallback: str) -> str:
        """Extract title from first H1 heading or use fallback."""
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        return fallback.replace('-', ' ').replace('_', ' ').title()

    def remove_first_h1(self, content: str) -> str:
        """Remove first H1 heading to avoid duplication with frontmatter title."""
        lines = content.split('\n')
        result = []
        removed = False
        for line in lines:
            if not removed and line.strip().startswith('# ') and not line.strip().startswith('## '):
                removed = True
                continue
            result.append(line)
        return '\n'.join(result)

    def fix_latex_math(self, content: str) -> str:
        """Fix LaTeX delimiters outside code blocks."""
        lines = content.split('\n')
        in_code = False
        result = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('```'):
                in_code = not in_code
                result.append(line)
                continue

            if in_code:
                result.append(line)
                continue

            # Fix triple $ to double $
            fixed = re.sub(r'\${3}([\s\S]*?)\${3}', r'$$\1$$', line)
            # Convert \[ \] to $$ $$
            fixed = re.sub(r'\\\[([\s\S]*?)\\\]', r'$$\1$$', fixed)
            result.append(fixed)

        return '\n'.join(result)

    def process_images(self, markdown: str, notebook_stem: str, output_dir: Path) -> str:
        """Extract base64 images and fix paths."""
        assets_dir = output_dir / notebook_stem
        assets_dir.mkdir(exist_ok=True)

        img_counter = [0]

        def save_base64_image(match):
            data_url = match.group(1)
            fmt = 'png' if 'image/png' in data_url else 'jpg'

            try:
                b64_data = data_url.split(',')[1]
                img_data = base64.b64decode(b64_data)

                img_counter[0] += 1
                filename = f"output_{img_counter[0]}.{fmt}"
                img_path = assets_dir / filename

                with open(img_path, 'wb') as f:
                    f.write(img_data)

                # Return relative path for Jekyll
                return f'![{fmt}]({notebook_stem}/{filename})'
            except Exception as e:
                logger.warning(f"Failed to process image: {e}")
                return match.group(0)

        # Handle base64 images in various formats
        markdown = re.sub(r"<img src='data:([^']+)'[^>]*>", save_base64_image, markdown)
        markdown = re.sub(r'!\[([^\]]*)\]\(data:([^)]+)\)',
                        lambda m: save_base64_image(type('', (), {'group': lambda s, i: m.group(2) if i==1 else ''})()),
                        markdown)

        return markdown

    def convert_notebook(self, lecture_info: dict) -> bool:
        """Convert a single notebook to markdown."""
        if not NOTEBOOK_SUPPORT:
            logger.error("nbconvert not available")
            return False

        notebook_path = lecture_info['notebook']
        output_path = lecture_info['path'] / "lecture.md"

        try:
            # Read notebook
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)

            # Convert to markdown
            exporter = MarkdownExporter()
            markdown, resources = exporter.from_notebook_node(notebook)

            # Process images from resources
            if resources.get('outputs'):
                assets_dir = lecture_info['path'] / "lecture"
                assets_dir.mkdir(exist_ok=True)
                for filename, data in resources['outputs'].items():
                    with open(assets_dir / filename, 'wb') as f:
                        f.write(data)
                # Fix image paths
                markdown = re.sub(
                    r'!\[([^\]]*)\]\(([^)]+)\)',
                    lambda m: f'![{m.group(1)}](lecture/{m.group(2)})',
                    markdown
                )

            # Process inline base64 images
            markdown = self.process_images(markdown, "lecture", lecture_info['path'])

            # Fix LaTeX
            markdown = self.fix_latex_math(markdown)

            # Extract title
            title = self.extract_title(markdown, lecture_info['lecture'])
            markdown = self.remove_first_h1(markdown)

            # Build Colab URL if configured
            colab_url = ""
            if self.github_repo:
                rel_path = notebook_path.relative_to(self.website_dir)
                colab_url = f"https://colab.research.google.com/github/{self.github_repo}/blob/{self.github_branch}/{rel_path}"

            # Create frontmatter
            frontmatter = f"""---
title: "{title}"
layout: note
module: "{lecture_info['module'].replace('-', ' ').title()}"
category: "Lecture"
notebook_source: "lecture.ipynb"
colab_url: "{colab_url}"
permalink: /lectures/{lecture_info['module']}/{lecture_info['lecture']}/
---

"""

            # Write output
            if not self.dry_run:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(frontmatter + markdown)
                logger.info(f"Converted: {lecture_info['module']}/{lecture_info['lecture']}")
            else:
                logger.info(f"[DRY RUN] Would convert: {lecture_info['module']}/{lecture_info['lecture']}")

            return True

        except Exception as e:
            logger.error(f"Failed to convert {notebook_path}: {e}")
            return False

    def sync_all(self):
        """Sync all lectures."""
        lectures = self.find_lectures()

        if not lectures:
            logger.warning(f"No lectures found in {self.lectures_dir}")
            return

        logger.info(f"Found {len(lectures)} lectures")

        converted = 0
        for lecture in lectures:
            if self.convert_notebook(lecture):
                converted += 1

        logger.info(f"Converted {converted}/{len(lectures)} lectures")


def main():
    parser = argparse.ArgumentParser(description="Sync SCDS lecture notebooks to markdown")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--base-dir", default=".", help="Website base directory")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    website_dir = Path(args.base_dir)
    if not (website_dir / "_config.yml").exists():
        # Try parent if we're in scripts/
        website_dir = Path(args.base_dir).parent

    print("=" * 50)
    print(" SCDS Lecture Sync")
    print("=" * 50)

    if args.dry_run:
        print(" [DRY RUN MODE]")

    sync = LectureSync(website_dir, dry_run=args.dry_run)
    sync.sync_all()

    print("\nDone!")


if __name__ == "__main__":
    main()
