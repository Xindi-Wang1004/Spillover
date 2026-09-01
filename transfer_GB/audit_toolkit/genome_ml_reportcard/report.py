from __future__ import annotations

from pathlib import Path


def write_markdown_report(report: dict, path: Path) -> None:
    cols = report.get("columns") or {}
    geom = report.get("geometry") or {}
    lines = [
        f"# GenomeML Report Card v{report.get('version', '?')}",
        "",
        f"- rows: {report.get('n_rows')}",
        f"- label-assignment groups: {report.get('n_groups')}",
        f"- label unit column: `{cols.get('label_assignment_unit')}`",
        f"- blocking unit column: `{cols.get('blocking_unit')}`",
        "",
        "## Label geometry",
        "",
        f"- within-block homogeneity ({geom.get('metric_type')}): {geom.get('within_block_homogeneity')}",
        f"- within-label-unit homogeneity: {geom.get('within_label_unit_homogeneity')}",
        f"- n blocks: {(geom.get('block_stats') or {}).get('n_blocks')}",
        f"- median block size: {(geom.get('block_stats') or {}).get('median_block_size')}",
        f"- % singleton blocks: {(geom.get('block_stats') or {}).get('pct_singleton_blocks')}",
        f"- random-CV shared-block fraction: {geom.get('random_cv_shared_block_fraction')}",
        "",
    ]
    if report.get("overlap"):
        ov = report["overlap"]
        lines += [
            "## Overlap audit",
            "",
            f"- accession overlap: {ov.get('accession_overlap')}",
            f"- block overlap: {ov.get('block_overlap')}",
            "",
        ]
    probe = report.get("probe")
    if probe:
        primary = probe.get("primary_metric", "rho")
        lines += [
            "## Split-design contrast",
            "",
            f"- primary metric: {primary}",
            f"- random: {probe['random'].get(primary)}",
            f"- blocked: {probe['blocked'].get(primary)}",
            f"- Δ: {probe.get('delta')}",
            f"- Δρ (always reported): {probe.get('delta_rho')}",
            f"- n_blocks: {probe.get('n_groups')}",
            f"- frac singleton blocks: {probe.get('frac_singleton_groups')}",
            "",
        ]
        for w in probe.get("warnings") or []:
            lines.append(f"- **WARNING:** {w}")
        lines.append("")
    path.write_text("\n".join(lines))
