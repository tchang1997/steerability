import numpy as np
from rich import print
from rich.panel import Panel
from rich.table import Table
from rich.console import Console, Group

from steerability.steerability_metrics import get_dist_to_goal
from steerability.utils.config_utils import has_negprompt

from typing import Optional

# Some helpers initially used for post-processing results during plotting.

STEERING_GOALS = ["reading_difficulty", "textual_diversity", "text_length", "formality"]

def extract_oracle_best(df, mode="best", steering_goals=STEERING_GOALS):
    target_goals = [f"target_{goal}" for goal in steering_goals] # need to also groupby target goals to avoid duplicate prompt issues!
    grouping = df.fillna(-1).groupby(["text"] + target_goals)
    if mode == "best":
        best_completions = grouping.apply(lambda group: group.iloc[get_dist_to_goal(group, steering_goals=steering_goals).argmin()]) \
            .reset_index(drop=True)
    elif mode == "worst":
        best_completions = grouping.apply(lambda group: group.iloc[get_dist_to_goal(group, steering_goals=steering_goals).argmax()]) \
            .reset_index(drop=True)
    elif mode == "median":
        def argmedian(group):
            target_dist = get_dist_to_goal(group, steering_goals=steering_goals)
            med = target_dist.median()
            med_dist = np.abs(target_dist - med)
            return med_dist.argmin()
        
        best_completions = grouping.apply(lambda group: argmedian(group)) \
            .reset_index(drop=True)
    else:
        raise ValueError()
    return best_completions


def add_run_info_to_stats(cfg: dict, judge_cfg: dict, steer_stats: dict):
    run_info = {
        'model_id': cfg['model_id'],
        'prompt_strategy': cfg['prompt_strategy'],
        'negprompt': has_negprompt(cfg),
        'probe_path': cfg['probe'],
    }
    if judge_cfg is not None:
        run_info['judge'] = judge_cfg['model']
    steer_stats['run'] = run_info
    return steer_stats

def print_steerability_summary(steer_stats: dict, max_panel_width: Optional[int] = 60, stdout: Optional[bool] = True, return_html: Optional[bool] = False):
    total_responses = steer_stats["data"]["n_total"]

    # === Header Panel ===
    header_lines = [
        f"[bold]Model:[/bold] {steer_stats['run']['model_id']}",
        f"[bold]Prompting strategy:[/bold] {steer_stats['run']['prompt_strategy']}" +
        (" [dim](w/ neg. prompt)[/dim]" if steer_stats['run']['negprompt'] else ""),
        f"[bold]Probe:[/bold] {steer_stats['run']['probe_path']}",
        f"[bold]# of total responses:[/bold] {total_responses}",
    ]
    header = Panel("\n".join(header_lines), title="STEERABILITY REPORT", border_style="cyan", width=max_panel_width)

    # === Judge Panel ===
    judge_panel = None
    if 'judge' in steer_stats['run']:
        def pct(n): return f"{n} ({n / total_responses * 100:.2f}%)"

        judge_lines = [
            f"[bold]Judge model:[/bold] {steer_stats['run']['judge']}",
            f"[bold]# of valid responses:[/bold] {pct(steer_stats['data']['n_grounded'])}",
            f"[bold]# of LLM-flagged responses:[/bold] {pct(steer_stats['data']['n_flagged'])}",
            f"[bold]# of overruled responses:[/bold] {pct(steer_stats['data']['n_overruled'])}",
        ]
        judge_panel = Panel("\n".join(judge_lines), title="INTERACTIVE JUDGING", border_style="magenta", width=max_panel_width)

    # === Metrics Table ===
    table = Table(title="STEERABILITY METRICS", show_header=True, header_style="bold cyan", width=max_panel_width)
    table.add_column("Metric", style="bold")
    table.add_column("Median (IQR)", justify="right")

    def fmt(metric):
        m = steer_stats["steerability"][metric]["median"]
        s = steer_stats["steerability"][metric]["iqr"]
        return f"{m:.3f} ({s:.3f})"

    table.add_row("Steering Error", fmt("steering_error"))
    table.add_row("Miscalibration", fmt("miscalibration"))
    table.add_row("Orthogonality", fmt("orthogonality"))

    # === Final stacked layout ===
    elements = [header]
    if judge_panel:
        elements.append(judge_panel)
    elements.append(table)
    group = Group(*elements)
    if stdout:
        print(group)
    else:
        console = Console(record=True, width=max_panel_width, force_terminal=True, color_system="truecolor") # ANSI colors
        console.print(group)
        if return_html:
            html = console.export_html(inline_styles=True, code_format='<div class="steer-output" style="{stylesheet}">{code}</div>')
            return html
        else:
            ansi = console.export_text() # return the string instead 
            return ansi




