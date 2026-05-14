---
name: notes-from-photo
description: >
  Convert photos of handwritten notes (any subject — math, ML, neural networks,
  algorithms, physics, etc.) into a polished .md file inside D:\dev\notes\.
  Use this skill whenever the user sends one or more photos of handwritten notes
  and asks to digitize, convert, transcribe, or add them to the notes repo.
  Also trigger when the user says things like "сделай конспект", "переведи в md",
  "добавь в заметки", "оформи конспект по фото", or uploads lecture/notebook photos.
---

## Goal

Read the handwritten content from the photos, find or create the right `.md` file
in `D:\dev\notes\`, and write a clean, complete note that looks like it was typed
by an expert — not a transcription dump.

---

## Step 1 — Locate the target file

1. Identify the topic from the photos (e.g. SVM, backpropagation, Fourier series).
2. Map it to the folder tree: `D:\dev\notes\<domain>\<subtopic>\`.
   - Common domains: `ml/ml/`, `ml/nn/`, `math/`, `algorithms/`, `fe/`, `be/`, etc.
3. If a relevant `.md` file already exists there, **open and read it** before writing
   anything — you must preserve every line currently in it.
4. If no file exists, create one following the kebab-case naming pattern of siblings
   (e.g. `3-kernel-methods.md`).

---

## Step 2 — Transcribe and enrich

Work through the photos left-to-right, top-to-bottom. For each concept:

- Convert every formula to LaTeX. Inline math: `$...$`. Display math: `$$...$$`.
  Never write math as plain text (no `omega`, no `sum_i`, no `||w||`).
- **After every display formula** add a one-line explanation of each symbol that
  hasn't been defined yet in the current paragraph, using the pattern:
  `где $x$ — ..., $\omega$ — ..., $k$ — ...`
  Skip symbols that are self-evident from context or were just introduced in the
  preceding sentence.
- Write **flowing prose paragraphs**, not bullet-point lists of definitions.
  A note should read like a concise textbook passage, not a PowerPoint slide.
- If the handwriting skips a step, a proof, or a standard result that belongs
  here, add it. The goal is a self-contained note, not a literal transcript.
- Do **not** add headers, horizontal rules, or bold section titles unless the
  existing file already uses them. Match the file's existing style.
- **Exception — advantages/disadvantages and comparisons**: use a plain bullet
  list, not prose. Format exactly like this (no bold markers, no sub-bullets):

  ```
  Преимущества:

  * ...
  * ...

  Недостатки:

  * ...
  * ...
  ```

  The same pattern applies when comparing several methods side-by-side
  (e.g. L1 vs L2 vs ElasticNet) — one bullet per method, no prose sentences.
- Preserve all code blocks in the existing file exactly. Append new code only
  if the photos contain code or pseudocode.

---

## Step 3 — Generate images (when diagrams are present)

If the photos contain diagrams, graphs, or architecture drawings, reproduce them
as matplotlib `.png` files.

**Image location rule:** all images for a topic go into an `images/` subfolder
inside the same directory as the `.md` file. Always create it if it doesn't exist:

```
D:\dev\notes\ml\ml\svm\
  1-svm-lin-kernel.md
  images/
    svm-margin.png
    svm-hinge-loss.png
```

Save the generated script as `_gen_temp.py` in the topic directory, run it, then delete it.
Always use this exact fixed name — never invent a custom filename:

```python
# save to: <topic-dir>/_gen_temp.py  →  run  →  delete
plt.savefig('images/name.png', dpi=140, bbox_inches='tight', transparent=True)
```

**Every image must be dark-theme compatible:**

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

WHITE = '#FFFFFF'
GRAY  = '#AAAAAA'
DIM   = '#555555'

os.makedirs('images', exist_ok=True)

fig = plt.figure(figsize=(w, h))
fig.patch.set_facecolor('none')          # transparent figure background

ax = fig.add_subplot(111)
ax.set_facecolor('none')                 # transparent axes background
ax.tick_params(colors=WHITE)
ax.xaxis.label.set_color(WHITE)
ax.yaxis.label.set_color(WHITE)
ax.title.set_color(WHITE)
for spine in ax.spines.values():
    spine.set_edgecolor(DIM)
ax.grid(color=DIM, alpha=0.35, linewidth=0.8)

# ... draw content with white/light-colored lines and text ...

plt.savefig('images/name.png', dpi=140, bbox_inches='tight', transparent=True)
```

Use light accent colors for data series so they read on dark backgrounds:
`#64B5F6` (blue), `#EF9A9A` (red), `#A5D6A7` (green), `#FFF176` (yellow),
`#FFCC80` (orange), `#CE93D8` (purple).

If images already exist in `images/`, **regenerate** them with the
transparent/white style above — do not leave old opaque versions.

Reference images in the `.md` file with the path `images/name.png`:

```markdown
![caption](images/name.png)
```

Place each image immediately before the paragraph it illustrates.

---

## Step 4 — Write / update the file

- If the file already existed: **append or insert** new content only.
  Never delete or rewrite sections that were already there.
- If the file is new: write the full note from scratch using the prose style above.
- After writing, do a quick self-check:
  - All math in LaTeX?
  - No plain-text "omega", "lambda", "||w||", etc.?
  - No excessive headers or bullet lists where prose would work?
  - Images generated and referenced?

---

## Conventions for this repo

- File names: kebab-case, optionally prefixed with a number (`2-svm-nonlinear-kernel.md`).
- Image names: kebab-case, descriptive (`kernel-trick.png`, `svm-margin.png`).
- Images always live in `images/` subfolder next to the `.md` file. Never put images directly in the topic folder.
- Reference path in `.md`: always `images/name.png`, never just `name.png`.
- Language of prose: **match the language already used in the target file**
  (usually Russian for this repo). LaTeX formulas are always language-neutral.
- No YAML frontmatter in notes files.
- The separator `---` followed by code blocks is acceptable at the end of a file
  to separate theory from practice examples.
