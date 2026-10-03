#!/usr/bin/env python3
"""Render the planning comparison using the established README visual system."""

from render_figures import Figure, INK, LINE, MONO, MUTED, PAPER, RUST, TEAL, WHITE


def planning_options():
    f = Figure(
        "04-planning-options", 828,
        "Planning over alternatives",
        "Illustrative next-action options in one planner response; one selected action passes runtime checks before dispatch.",
    )
    f.header("04", "Compare before committing.",
             "2–4 candidate next actions, one selected research decision.")
    f.text(54, 207, "RESEARCH CONTEXT", 16, MUTED, spacing=1.6)
    f.text(330, 207, "CANDIDATE NEXT ACTIONS", 16, MUTED, spacing=1.6)
    f.text(1386, 207, "ONE PLANNER CALL", 16, TEAL, anchor="end", spacing=1.4)

    f.rect(54, 230, 236, 366)
    f.text(76, 270, "Current study", 26, INK, weight="bold")
    f.text(76, 302, "EEG / MEG retrieval", 17, TEAL)
    f.path("M78 342 H101 L113 329 L126 361 L139 317 L151 350 L164 338 H264", TEAL, 2.0)
    for y, label in ((389, "Question + evidence"), (431, "Eligible targets"),
                     (473, "Remaining budget"), (515, "Frozen protocol")):
        f.circle(80, y - 5, 3, RUST)
        f.text(94, y, label, 17, INK)
    f.text(76, 563, "Development scope", 16, MUTED)

    f.rect(316, 230, 686, 366, WHITE, LINE)
    rows = [
        ("A", "Read registered evidence", "Resolve a specific missing record or page.", False),
        ("B", "Diagnose a result", "Distinguish an unresolved decoding explanation.", False),
        ("C", "Design a new experiment", "Specify a controlled encoder or loss change.", True),
        ("D", "Replicate an eligible Full run", "Follow the frozen seed and confirmation policy.", False),
    ]
    for index, (letter, title, detail, selected) in enumerate(rows):
        y = 242 + index * 85
        f.rect(330, y, 658, 73, "#EAF2EE" if selected else PAPER,
               TEAL if selected else LINE, radius=5, width=1.6 if selected else 1)
        f.circle(354, y + 31, 13, TEAL if selected else WHITE, TEAL if selected else LINE, 1)
        f.text(354, y + 37, letter, 16, WHITE if selected else RUST, font=MONO, anchor="middle")
        f.text(380, y + 29, title, 22, TEAL if selected else INK, weight="bold")
        f.text(380, y + 54, detail, 16, MUTED)

    f.path("M290 414 H316", RUST, 1.8, arrow="rust")
    f.path("M1002 448 H1042", TEAL, 2.0, arrow="teal")
    f.rect(1042, 230, 344, 366, WHITE, TEAL, radius=5, width=1.6)
    f.text(1064, 260, "SELECTED OPTION / EXAMPLE", 14, TEAL, spacing=1)
    f.text(1064, 300, "Design experiment", 26, INK, weight="bold")
    f.text(1064, 333, "Target, evidence and", 18, MUTED)
    f.text(1064, 359, "selection rationale recorded.", 18, MUTED)
    f.rule(1064, 382, 1364)
    f.text(1064, 415, "Runtime validates", 22, INK, weight="bold")
    f.text(1064, 445, "Legal action / eligible target", 17, MUTED)
    f.text(1064, 470, "Evidence / required gates", 17, MUTED)
    f.text(1064, 495, "Remaining budget", 17, MUTED)
    f.rect(1064, 523, 300, 51, TEAL, TEAL, radius=5)
    f.text(1214, 556, "Dispatch one action", 22, WHITE, weight="bold", anchor="middle")

    f.path("M1214 596 V617 H172 V596", TEAL, 1.6, dash=True, arrow="teal")
    f.rect(517, 603, 405, 28, PAPER, PAPER)
    f.text(720, 623, "Update the study record", 18, TEAL, anchor="middle")
    f.rect(54, 650, 1332, 85)
    f.text(76, 687, "After the action", 24, INK, weight="bold")
    f.text(336, 687, "Observed evidence informs the next planning decision.", 22, TEAL)
    f.text(336, 717, "Compare evidence gaps, expected information, prerequisites and resource estimates.", 17, MUTED)
    f.footer("Illustrative planning example. Usually 2–4 options; one is allowed when only one is reasonable.")
    f.save()


if __name__ == "__main__":
    planning_options()
    print("Rendered planning options as SVG and a 2160px PNG.")
