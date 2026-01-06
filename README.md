# SCDS Course Materials

Lecture materials for the School of Data Science and Computing at the American University of Beirut.

## Live Site

Visit: https://scds-aub.github.io/course-materials/

## Structure

```
website/
├── index.html              # Homepage
├── lectures.html           # All lectures listing
├── analysis.html           # Faculty research analysis
├── lectures/
│   └── vision/            # Vision module
│       ├── how-machines-see-1/
│       │   ├── transcript.md
│       │   ├── lecture.md
│       │   ├── lecture.ipynb
│       │   └── lecture.html
│       └── how-machines-see-2/
│           └── ...
├── scripts/
│   └── generate_lecture.py # Lecture generation script
├── _layouts/               # Jekyll templates
└── assets/css/            # Stylesheets
```

## Adding New Lectures

1. Create folder: `lectures/<module>/<lecture-name>/`
2. Add transcript: `transcript.md`
3. Run generator:
   ```bash
   python scripts/generate_lecture.py <module> <lecture-name> --transcript transcript.md
   ```
4. Edit generated `lecture.md` with full content
5. Commit and push

## Local Development

```bash
# Install dependencies
bundle install

# Run local server
bundle exec jekyll serve

# Visit http://localhost:4000
```

## Content Pipeline

1. **Transcript** - Voice notes or written outline
2. **Notebook** - Interactive Jupyter notebook
3. **Markdown** - Web-ready formatted content
4. **Website** - Published for students

## Modules

- [x] Vision - Computer vision fundamentals
- [ ] Language - NLP and text processing
- [ ] Time Series - Temporal data analysis
- [ ] Networks - Graph analysis
- [ ] Statistics - Probabilistic thinking
- [ ] Audio - Signal processing

## Contributing

See course coordinator for content contributions.

---

School of Data Science and Computing, AUB
# intro2ds
