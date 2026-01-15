# SCDS Intro to Data Science Website

Course materials website for DATA 201/202 at AUB School of Data Science and Computing.

**Live site:** https://scds-aub.github.io/intro2ds/

## Structure

```
website/
├── _config.yml          # Jekyll configuration
├── _layouts/            # Page templates (default, note, lecture)
├── lectures/            # Lecture content (ipynb + generated md)
├── scripts/
│   ├── sync_lectures.py         # Convert ipynb → md
│   └── find_collaborators.py    # Search AUB publications
├── data/collaborators.json      # Faculty-module mapping
├── syllabus.md          # Full curriculum with collaborators
├── collaborators.md     # Faculty by department
└── .github/workflows/jekyll.yml # GitHub Actions deployment
```

## Local Development

```bash
bundle install
bundle exec jekyll serve --port 4001
# View at http://127.0.0.1:4001/intro2ds/
```

## Adding Lectures

1. Create folder: `lectures/<module>/<lecture-name>/`
2. Add `lecture.ipynb`
3. Run: `python scripts/sync_lectures.py`

## Collaborator Search

```bash
python scripts/find_collaborators.py
```
Searches AUB ScholarWorks (2021+) for faculty matching module topics.

## Features

- MathJax LaTeX rendering
- GitHub Dark syntax highlighting
- Colab badge integration
- Giscus comments (GitHub Discussions)

## Deployment

GitHub Actions auto-deploys on push to main.
Settings → Pages → Source: "GitHub Actions"
