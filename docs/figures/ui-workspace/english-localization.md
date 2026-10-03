# English localization of the recorded workbench image

Source: `ui-workspace-1440.png`, committed with `14ec3e6` on 2026-10-01.

Output: `ui-workspace-1440-en.png`. Created with the built-in imagegen text-localization workflow for documentation. This is an edited image, not a newly captured application view. The original capture remains available.

The edit translates interface labels, role names, statuses and visible execution steps to English. The central steps are: Created → LLM call started → LLM call finished → Inspect data → Retrieve method cards → Design experiment → Experiment design → Implement candidate code → Repair candidate → Run pilot.

Prompt constraints: translate all visible Chinese text into accurate English; preserve the three-column layout, warm paper / ink / teal / copper palette, campaign identifiers, synthetic values, timestamps, event order and recorded statuses; retain the DEMO label and failed stop; do not add results or curves. Small typography and spacing adjustments are permitted for the English text.

The exact edit prompt is included in `english-localization-prompt.txt`.
