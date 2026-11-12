"""Describe saved test predictions without training or model selection.

Run from any directory. Regenerates the paired comparison and event contribution
CSVs in plots/, and the corresponding report tables. No additional inference.
"""
import csv
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = 'maestro-dropout02-best-test-20261107-v1'


def read_rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze():
    source = ROOT / 'results' / RUN
    output = ROOT / 'plots' / RUN
    output.mkdir(parents=True, exist_ok=True)
    pieces, events = {}, {}
    for size in ('small', 'medium', 'large'):
        piece_rows = read_rows(source / size / 'test_per_piece.csv')
        pieces[size] = {row['midi_filename']: row for row in piece_rows}
        if len(pieces[size]) != len(piece_rows):
            raise ValueError(f'{size}: duplicate piece identities')
        events[size] = {row['event_family']: row for row in read_rows(source / size / 'test_by_event.csv')}
    reference = pieces['small']
    total = sum(int(row['scored_tokens']) for row in reference.values())
    paired, contributions = [], []
    summary = {row['model']: row for row in read_rows(source / 'comparison.csv')}
    for size in ('small', 'medium', 'large'):
        current = pieces[size]
        if current.keys() != reference.keys():
            raise ValueError(f'{size}: piece identities differ')
        for key in reference:
            if current[key]['scored_tokens'] != reference[key]['scored_tokens']:
                raise ValueError(f'{size}: token count differs for {key}')
        weighted = sum(int(row['scored_tokens']) * float(row['cross_entropy']) for row in current.values()) / total
        if not math.isclose(weighted, float(summary[size]['test_loss']), abs_tol=1e-8):
            raise ValueError(f'{size}: per-piece losses do not reproduce the overall loss')
        delta = [float(current[key]['cross_entropy']) - float(reference[key]['cross_entropy']) for key in reference]
        paired.append(dict(model=size, pieces=len(current), token_weighted_loss=weighted,
                           equal_piece_loss=statistics.mean(float(row['cross_entropy']) for row in current.values()),
                           pieces_with_higher_loss_than_small=sum(x > 0 for x in delta),
                           median_loss_minus_small=statistics.median(delta)))
        if size == 'small':
            continue
        if events[size].keys() != events['small'].keys():
            raise ValueError(f'{size}: event families differ')
        model_contributions = []
        for family, row in events[size].items():
            baseline = events['small'][family]
            if row['scored_tokens'] != baseline['scored_tokens']:
                raise ValueError(f'{size}: event counts differ')
            share = int(row['scored_tokens']) / total
            value = share * (float(row['cross_entropy']) - float(baseline['cross_entropy']))
            contributions.append(dict(model=size, event_family=family, target_share=share,
                                      contribution_to_loss_minus_small=value))
            model_contributions.append(value)
        if not math.isclose(sum(model_contributions), weighted - paired[0]['token_weighted_loss'], abs_tol=1e-8):
            raise ValueError(f'{size}: event contributions do not reproduce the loss difference')
    write_rows(output / 'paired_piece_comparison.csv', paired)
    write_rows(output / 'event_loss_contributions.csv', contributions)
    tables = ROOT / 'report' / 'tables'
    tables.mkdir(exist_ok=True)
    table = [r'\begin{table}[H]', r'\centering',
             r'\caption{Exploratory checks on the same saved test predictions. Equal-piece loss gives every piece the same weight. The last column counts pieces where small has lower loss. These are not repeated training runs.}',
             r'\label{tab:pairedtest}', r'\begin{tabular}{lrrr}',
             r'\toprule Model & Token-weighted loss & Equal-piece loss & Small wins / 177\\\midrule']
    for row in paired:
        wins = '--' if row['model'] == 'small' else str(row['pieces_with_higher_loss_than_small'])
        table.append(f"{row['model'].capitalize()} & {row['token_weighted_loss']:.4f} & {row['equal_piece_loss']:.4f} & {wins}" + r'\\')
    table += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (tables / 'paired_test_summary.tex').write_text('\n'.join(table) + '\n')
    family_table = [r'\begin{table}[H]', r'\centering',
                    r'\caption{Per-event-family test cross-entropy for the selected configurations. Counts are shared across models.}',
                    r'\label{tab:eventdiagnostics}', r'\begin{tabular}{lrrrr}',
                    r'\toprule Family & Targets & Small ($p=0.1$) & Medium ($p=0.2$) & Large ($p=0.2$)\\\midrule']
    for family, row in events['small'].items():
        label = family.replace('_', r'\_')
        losses = [f"{float(events[size][family]['cross_entropy']):.4f}" for size in ('small', 'medium', 'large')]
        family_table.append(f"{label} & {int(row['scored_tokens']):,} & " + ' & '.join(losses) + r'\\')
    family_table += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (tables / 'event_family_losses.tex').write_text('\n'.join(family_table) + '\n')
    for row in paired:
        print(f"{row['model']}: equal-piece loss {row['equal_piece_loss']:.4f}; small wins {row['pieces_with_higher_loss_than_small']}/{row['pieces']}")


if __name__ == '__main__':
    analyze()
