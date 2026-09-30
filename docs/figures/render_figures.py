#!/usr/bin/env python3
"""Render EEGagent's editable, data-free README illustrations.

Run from any directory with Python 3 and Inkscape installed. No model, API,
training data, or third-party Python package is required.
"""

from __future__ import annotations

import html
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
PAPER = "#F7F4EE"
INK = "#203744"
MUTED = "#5C6C70"
LINE = "#CCCFC7"
TEAL = "#35756E"
RUST = "#AE663B"
WHITE = "#FFFDFA"
SANS = "DejaVu Sans"
SERIF = "DejaVu Serif"
MONO = "DejaVu Sans Mono"


class Figure:
    def __init__(self, name: str, height: int, title: str, description: str):
        self.name, self.height = name, height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="{height}" '
            f'viewBox="0 0 1440 {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{html.escape(title)}</title>',
            f'<desc id="desc">{html.escape(description)}</desc>',
            '<defs>',
            *[
                f'<marker id="arrow-{name}" markerWidth="9" markerHeight="9" refX="7" '
                f'refY="4" orient="auto" markerUnits="userSpaceOnUse">'
                f'<path d="M0 0 L8 4 L0 8" fill="{color}"/></marker>'
                for name, color in (("ink", INK), ("teal", TEAL), ("rust", RUST))
            ],
            '</defs>',
            f'<rect width="1440" height="{height}" fill="{PAPER}"/>',
        ]

    def text(self, x, y, label, size=24, color=INK, font=SANS, weight="normal", anchor="start", spacing=0):
        self.parts.append(
            f'<text x="{x}" y="{y}" fill="{color}" font-family="{font}" '
            f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
            f'letter-spacing="{spacing}">{html.escape(label)}</text>'
        )

    def rect(self, x, y, w, h, fill=WHITE, stroke=LINE, radius=0, width=1.2):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def path(self, d, color=INK, width=2, fill="none", dash=False, arrow=None, opacity=1):
        self.parts.append(
            f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{width}" '
            f'stroke-linecap="round" stroke-linejoin="round" opacity="{opacity}"'
            + (' stroke-dasharray="7 7"' if dash else '')
            + (f' marker-end="url(#arrow-{arrow})"' if arrow else '') + '/>'
        )

    def circle(self, x, y, r, fill=TEAL, stroke="none", width=1):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def rule(self, x1, y, x2, color=LINE):
        self.path(f'M{x1} {y} H{x2}', color, 1)

    def header(self, index, title, subtitle):
        self.text(54, 47, "EEGagent  /  BRAIN DECODING RESEARCH", 16, MUTED, spacing=2)
        self.text(1385, 47, index, 18, RUST, font=MONO, anchor="end")
        self.text(54, 109, title, 43, font=SERIF)
        self.text(54, 150, subtitle, 23, MUTED)

    def footer(self, text):
        self.rule(54, self.height - 70, 1386)
        self.text(54, self.height - 33, text, 18, MUTED)

    def save(self):
        path = ROOT / f'{self.name}.svg'
        path.write_text('\n'.join(self.parts + ['</svg>']) + '\n', encoding='utf-8')
        subprocess.run([
            'inkscape', str(path), '--export-type=png', '--export-width=2160',
            f'--export-filename={ROOT / (self.name + ".png")}',
        ], check=True, capture_output=True)


def brain(f, x, y, scale=1, electrodes=False, activity=False):
    """A stylized lateral hemisphere; no anatomical or measured-data claim."""
    f.parts.append(f'<g transform="translate({x} {y}) scale({scale})">')
    f.path('M22 111 C2 90 8 54 30 43 C32 20 57 10 78 17 C96 0 126 5 139 18 '
           'C165 10 190 24 193 47 C220 58 224 90 209 107 C211 134 182 151 157 146 '
           'L136 151 C117 155 108 138 111 124 C90 132 71 125 57 127 C39 130 27 122 22 111 Z',
           INK, 2.6, '#EBEEE5')
    folds = [
        'M32 48 C58 31 73 47 71 67 C67 84 43 73 43 94 C43 109 62 114 77 103',
        'M80 22 C97 43 89 54 103 67 C112 77 125 67 126 54 C128 36 145 30 162 39',
        'M45 38 C48 24 65 22 77 29',
        'M111 21 C123 31 113 46 106 49',
        'M177 47 C158 49 149 65 161 78 C169 87 193 74 202 91',
        'M82 76 C100 73 110 88 99 102 C91 114 75 115 79 122',
        'M116 94 C135 83 147 89 146 106 C145 126 167 132 188 117',
        'M164 89 C181 88 191 101 184 111',
        'M118 124 C131 116 138 131 140 142',
    ]
    for d in folds:
        f.path(d, TEAL, 1.7, opacity=.65)
    if activity:
        for cx, cy, r in ((40, 87, 10), (59, 100, 7), (167, 59, 8), (180, 104, 11)):
            f.circle(cx, cy, r, RUST, width=0)
    if electrodes:
        for cx, cy in ((34, 36), (73, 12), (113, 10), (161, 22), (199, 57)):
            f.circle(cx, cy, 6, PAPER, RUST, 2)
    f.parts.append('</g>')


def traces(f, x, y, w=250, rows=3, amp=12):
    for row in range(rows):
        baseline = y + row * 32
        f.path(f'M{x} {baseline} H{x+w}', LINE, .8)
        points = []
        for k in range(121):
            v = (math.sin(k * .28 + row * .8) * .46 + math.sin(k * .71 + row) * .22
                 + math.sin(k * .13 + row) * .3)
            points.append(f'{x+k*w/120:.2f} {baseline-amp*v:.2f}')
        f.path('M' + ' L'.join(points), TEAL if row % 2 == 0 else RUST, 1.9)


def embedding(f, x, y, cell=12):
    for row in range(5):
        for col in range(8):
            f.rect(x + col * (cell+4), y + row * (cell+4), cell, cell,
                   (TEAL, '#B8CEC4', '#DEE6DC', RUST, '#E6C8AF')[(row*3+col*2) % 5], 'none')


def gallery(f, x, y, width=63, rows=1, highlight=True):
    for row in range(rows):
        for col in range(3):
            xx, yy = x+col*(width+14), y+row*(width+14)
            color = RUST if highlight and col == 1 and row == 0 else LINE
            f.rect(xx, yy, width, width, '#EEEAE1', color, width=2 if color == RUST else 1)
            f.circle(xx+width*.72, yy+width*.23, width*.08, '#E3C39D')
            f.path(f'M{xx+6} {yy+width-8} L{xx+width*.32} {yy+width*.45} '
                   f'L{xx+width*.52} {yy+width*.65} L{xx+width*.72} {yy+width*.42} '
                   f'L{xx+width-6} {yy+width-8} Z', 'none', 0, '#A7B9AA')


def hero():
    f = Figure('00-brain-decoding-hero', 670, 'EEGagent: from brain signals to testable ideas',
               'Editorial illustration of EEG signals, learned representations and image-gallery retrieval. '
               'The brain and signal traces are conceptual drawings, not recorded data or reconstruction results.')
    f.text(54, 53, 'BRAIN DECODING  /  AN EXPERIMENTAL WORKBENCH', 18, MUTED, spacing=2)
    f.rule(54, 76, 1386)
    f.text(50, 210, 'EEGagent', 108, font=SERIF)
    f.text(58, 274, 'From brain signals', 37, font=SERIF)
    f.text(58, 324, 'to testable ideas.', 37, font=SERIF)
    f.text(58, 376, 'Plan the experiment. Read the evidence. Decide what comes next.', 23, MUTED)
    brain(f, 1053, 119, 1.38, electrodes=True)
    traces(f, 965, 365, 364, rows=1, amp=16)
    f.text(1340, 412, 'Neural signals, drawn schematically', 17, MUTED, anchor='end')
    f.rule(54, 438, 1386)
    traces(f, 72, 495, 224, rows=2, amp=11)
    f.text(330, 501, '01  SIGNAL', 17, RUST, font=MONO)
    f.text(330, 538, 'EEG / MEG', 26, font=SERIF)
    f.path('M492 513 H548', MUTED, 1.5, arrow='ink')
    embedding(f, 579, 477, 10)
    f.text(731, 501, '02  REPRESENTATION', 17, RUST, font=MONO)
    f.text(731, 538, 'Learned alignment', 26, font=SERIF)
    f.path('M984 513 H1040', MUTED, 1.5, arrow='ink')
    gallery(f, 1070, 477, 55)
    f.text(1070, 568, '03  IMAGE RETRIEVAL', 17, RUST, font=MONO)
    f.footer('EEG / MEG retrieval research   ·   fMRI screening   /   Conceptual illustration')
    f.save()


def overview():
    f = Figure('01-system-overview', 800, 'Two studies. One research workbench.',
               'Separate EEG/MEG retrieval and fMRI screening workflows share a local workbench. '
               'Neither workflow automatically feeds into the other.')
    f.header('01', 'Choose the question you want to investigate.', 'One interface for two workflows, each with its own evidence and memory.')
    for x in (54, 746):
        f.rect(x, 197, 640, 516)
    f.text(80, 242, 'A  /  EEG + MEG', 18, RUST, font=MONO)
    f.text(80, 287, 'Brain-to-image retrieval', 32, font=SERIF)
    traces(f, 86, 343, 212, rows=3, amp=11)
    f.path('M329 375 H380', MUTED, 1.5, arrow='ink')
    gallery(f, 413, 337, 63)
    f.text(85, 456, 'Preprocessed signals + fixed image features', 22, MUTED)
    f.rule(80, 486, 668)
    f.text(80, 530, 'Ask what changes the decoder.', 25, weight='bold')
    for y, line in ((573, 'Design a control. Implement a candidate.'), (608, 'Train, compare and revise the next step.'), (663, 'Outputs  /  code · comparisons · checkpoints')):
        f.text(80, y, line, 22, MUTED)
    f.text(772, 242, 'B  /  fMRI', 18, TEAL, font=MONO)
    f.text(772, 287, 'Screen predicted responses', 32, font=SERIF)
    brain(f, 786, 316, .76, activity=True)
    f.path('M995 376 H1042', MUTED, 1.5, arrow='ink')
    for row in range(5):
        for col in range(7):
            f.rect(1081+col*25, 327+row*22, 22, 19,
                   '#D4DDD2' if (row+col)%3 else '#D4A88B', 'none')
    f.text(777, 456, 'Existing series, or configured TRIBE generation', 21, MUTED)
    f.rule(772, 486, 1360)
    f.text(772, 530, 'Ask which checks the response passes.', 25, weight='bold')
    for y, line in ((573, 'Run required checks. Inspect remaining questions.'), (608, 'Select a tool, update evidence, then stop or replan.'), (663, 'Outputs  /  screening report · diagnostics')):
        f.text(772, y, line, 21, MUTED)
    f.footer('Shared workbench and controls. Separate protocols, state and memory. All signal imagery is schematic.')
    f.save()


def role(f, x, y, name, description, color=INK):
    f.circle(x+5, y-8, 4, color)
    f.text(x+23, y, name, 25, color, weight='bold')
    f.text(x+23, y+29, description, 21, MUTED)


def collaboration():
    f = Figure('02-eeg-agent-collaboration', 1140, 'The brain decoding research loop',
               'Eight LLM roles are grouped by research stage around an EEG-to-image retrieval study. '
               'The campaign runtime routes every handoff. Arrows show representative stages, '
               'not fixed execution order, direct messages, or parallel autonomous processes.')
    f.header('02', 'Inside an EEG decoding study.', 'Eight roles, coordinated through campaign state, controls and recorded evidence.')
    brain(f, 1244, 64, .54, electrodes=True)
    f.rule(54, 173, 1386)
    f.text(54, 191, 'THE STUDY  /  neural signals → candidate representation → fixed image gallery', 16, TEAL, font=MONO)
    # Four panels create an explicit clockwise loop; the data illustration is inset.
    for x, y in ((54, 207), (785, 207), (785, 630), (54, 630)):
        f.rect(x, y, 601, 354)
    f.text(80, 248, '01  /  FRAME THE QUESTION', 17, RUST, font=MONO)
    role(f, 80, 291, 'Research Planner', 'Choose an action using evidence and budget.')
    role(f, 80, 374, 'Research Librarian', 'Bring local methods and applicability limits.')
    role(f, 80, 457, 'Experiment Designer', 'Turn a hypothesis into a controlled experiment.')
    f.rule(80, 516, 627)
    f.text(80, 544, 'Study contract  /  split · gallery · metric · seeds', 19, TEAL)
    f.text(811, 248, '02  /  BUILD THE CANDIDATE', 17, RUST, font=MONO)
    role(f, 811, 299, 'Candidate Coder', 'Implement an isolated extension.')
    role(f, 811, 396, 'Candidate Reviewer', 'Check the implementation; request repairs.')
    f.path('M1297 395 H1348 V298 H1310', RUST, 1.4, dash=True, arrow='rust')
    f.rule(811, 468, 1360)
    f.text(811, 506, 'Runtime approval binds the experiment context.', 21, TEAL)
    f.text(811, 540, 'Supported hooks + source and configuration identity', 19, MUTED)
    f.text(811, 671, '03  /  READ THE EVIDENCE', 17, RUST, font=MONO)
    role(f, 811, 717, 'Result Analyst', 'Interpret matched comparisons and diagnostics.')
    f.rule(811, 772, 1360)
    f.text(811, 811, 'Runtime  /  train → settle → compare', 25, TEAL, font=SERIF)
    f.text(811, 851, 'Baseline, pilot, full run or replication as selected.', 21, MUTED)
    f.text(811, 889, 'Fixed validation gallery. Declared confirmation policy.', 20, MUTED)
    f.text(811, 949, 'Findings return to planning — including negative ones.', 20, INK)
    f.text(80, 671, '04  /  RETAIN WHAT IS SUPPORTED', 17, RUST, font=MONO)
    role(f, 80, 723, 'Result Auditor', 'Check draft claims against the latest result.')
    role(f, 80, 819, 'Memory Curator', 'Propose lessons with conditions and evidence.')
    f.rule(80, 884, 627)
    f.text(80, 922, 'Runtime  /  audit freshness + lesson acceptance', 21, TEAL)
    f.text(80, 955, 'Records preserve scope, dependencies and limits.', 20, MUTED)
    # Stage handoffs and two feedback routes have clear spaces between panels.
    f.path('M655 319 H777', RUST, 2, arrow='rust')
    f.text(720, 292, 'draft', 17, MUTED, anchor='middle')
    f.path('M1085 561 V622', RUST, 2, arrow='rust')
    f.text(1105, 602, 'ready', 17, MUTED)
    f.path('M785 887 H663', RUST, 2, dash=True, arrow='rust')
    f.text(720, 858, 'findings', 17, MUTED, anchor='middle')
    f.path('M351 630 V569', TEAL, 2, dash=True, arrow='teal')
    f.text(330, 602, 'scoped memory', 17, MUTED, anchor='end')
    f.path('M785 743 H734 V382 H663', TEAL, 1.8, dash=True, arrow='teal')
    f.rect(54, 1015, 1332, 37, fill=INK, stroke=INK)
    f.text(720, 1041, 'CAMPAIGN RECORDS  /  versioned plans · attempts · jobs · comparisons · conditional lessons', 20, WHITE, anchor='middle')
    f.footer('Representative handoffs, mediated by the runtime. Roles are invoked as needed; pilot evidence remains pilot-level.')
    f.save()


def screening():
    f = Figure('03-fmri-screening-loop', 1020, 'Screen, inspect, and choose the next check',
               'An existing fMRI series or optional configured image-to-TRIBE generation enters required checks. '
               'A rule policy or configured LLM planner selects optional tools, runtime validation precedes execution, '
               'and evidence returns to planning. Reports retain numeric_consistency_only scope.')
    f.header('03', 'Which checks does this response pass?', 'A separate fMRI screening loop, built around measurements and open questions.')
    f.rect(54, 201, 380, 598)
    f.text(80, 245, 'INPUT  /  SURFACE RESPONSE', 17, TEAL, font=MONO)
    brain(f, 115, 280, 1.0, activity=True)
    f.text(80, 486, 'Existing fMRI series', 26, font=SERIF)
    f.text(80, 527, 'or image → TRIBE + gray control', 19, MUTED)
    f.text(80, 558, 'External resources required.', 19, MUTED)
    f.rule(80, 591, 408)
    f.text(80, 636, 'Ingest + required checks', 23, weight='bold')
    f.text(80, 679, 'Establish the initial evidence.', 21, MUTED)
    f.text(80, 744, 'Activity colors are schematic.', 18, MUTED)
    f.rect(516, 201, 399, 219)
    f.text(543, 245, 'PLAN', 17, RUST, font=MONO)
    f.text(543, 287, 'Select the next check', 29, font=SERIF)
    f.text(543, 329, 'Rule policy or configured LLM', 19, MUTED)
    f.text(543, 371, 'Read evidence and open questions.', 19, MUTED)
    f.rect(987, 201, 399, 219)
    f.text(1014, 245, 'VALIDATE + EXECUTE', 17, TEAL, font=MONO)
    f.text(1014, 287, 'Measure the response', 29, font=SERIF)
    f.text(1014, 329, 'Runtime validates each action first.', 20, MUTED)
    f.text(1014, 371, 'Run tools with available resources.', 19, MUTED)
    f.rect(987, 493, 399, 218)
    f.text(1014, 537, 'OBSERVE', 17, TEAL, font=MONO)
    f.text(1014, 579, 'Update the evidence', 29, font=SERIF)
    f.text(1014, 621, 'Measurements + question ledger', 21, MUTED)
    f.text(1014, 662, 'Continue, or finalize within policy.', 20, MUTED)
    f.rect(516, 493, 399, 218)
    f.text(543, 537, 'FINALIZE WHEN READY', 17, RUST, font=MONO)
    f.text(543, 579, 'Write the report', 29, font=SERIF)
    f.text(543, 618, 'Verdict + screening decision', 19, MUTED)
    f.text(543, 648, 'Stop reason + memory episode', 19, MUTED)
    f.text(543, 683, 'Optional curation of scoped lessons', 18, MUTED)
    f.path('M434 315 H508', RUST, 1.8, arrow='rust')
    f.path('M915 315 H979', RUST, 1.8, arrow='rust')
    f.path('M1186 420 V485', RUST, 1.8, arrow='rust')
    f.path('M987 602 H923', RUST, 1.8, arrow='rust')
    f.path('M1005 493 V457 H715 V428', TEAL, 1.8, dash=True, arrow='teal')
    f.text(805, 449, 'replan', 17, MUTED, anchor='middle')
    f.path('M715 711 V773 H478 V383 H508', TEAL, 1.4, dash=True, arrow='teal')
    f.text(731, 776, 'retrieve compatible experience', 17, MUTED)
    f.rule(516, 801, 1386)
    f.text(516, 837, 'CHECKS  /  numeric · temporal · surface & ROI · reference · specificity', 18, MUTED)
    f.text(516, 872, 'Optional semantic / CortexMAE checks depend on configuration and assets.', 18, MUTED)
    f.text(54, 925, 'REPORT SCOPE  /  numeric_consistency_only', 22, TEAL, font=MONO)
    f.footer('Screening evidence stays in the fMRI workflow. It does not become EEG training data automatically.')
    f.save()


if __name__ == '__main__':
    hero()
    overview()
    collaboration()
    screening()
    print('Rendered four SVG sources and four 2160px PNG exports.')
