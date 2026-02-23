#!/usr/bin/env python3
"""
Find all US RNAV (GPS) approaches where the +V advisory glidepath
violates the TERPS 250' minimum terrain clearance.

Usage:
    python approach_study.py              # all US approaches, console output
    python approach_study.py KSBS         # single airport, console output
    python approach_study.py --html       # all US approaches, HTML report
    python approach_study.py --html KSBS  # single airport, HTML report
"""

import argparse
import html
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import approaches as m
from profile import plot_approach_profile


def print_console(violations: "pd.DataFrame") -> None:
    """Print violations to console (original output format)."""
    print("\n" + "=" * 80)
    print(f"APPROACHES WITH +V GUIDANCE CLEARANCE < {m.TERPS_MIN_CLEARANCE_FT}'")
    print("=" * 80)

    if violations.empty:
        print("No violations found.")
    else:
        for row in violations.itertuples():
            print(f"  - {row.airport} {row.apch_name} "
                  f"fouls terrain at ({row.worst_lat:.6f}, {row.worst_lon:.6f}) "
                  f"@ {row.worst_clearance_ft}'")

    print(f"\nTotal: {len(violations)} approaches")


def generate_html(violations: "pd.DataFrame", db: m.ApproachDatabase,
                  output_dir: Path, srtm1: bool = False) -> None:
    """Generate HTML report with sortable table and per-approach profile charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    # Legs and terrain loaded lazily on first cache miss
    all_legs = None
    chart_terrain = None
    n_cached = 0

    n_violations = len(violations)
    table_rows = []

    for i, row in enumerate(violations.itertuples()):
        apt = row.airport
        proc_id = row.proc_id
        apch_name = row.apch_name
        clearance = row.worst_clearance_ft
        gpa = row.gpa_deg
        dist = row.distance_nm

        # Airport metadata
        info = db.get_airport_info(apt)
        apt_name = info["name"]
        state = info["state"]

        # Generate chart (skip if already exists)
        chart_filename = f"{apt}_{proc_id}.png"
        chart_path = charts_dir / chart_filename

        if chart_path.exists() and chart_path.stat().st_size > 0:
            n_cached += 1
        else:
            print(f"  [{i+1}/{n_violations}] {apt} {apch_name}...")
            if all_legs is None:
                print("Loading approach legs for chart generation...")
                all_legs = db.load_approach_legs()
                res_label = "SRTM1 30m" if srtm1 else "SRTM3 90m"
                print(f"Initializing terrain for profile charts ({res_label})...")
                chart_terrain = db.get_terrain(srtm1=srtm1)

            apt_legs = all_legs[all_legs["airport"] == apt]
            apch_legs = apt_legs[apt_legs["proc_id"] == proc_id]

            if not apch_legs.empty:
                fig, ax = plt.subplots(1, 1, figsize=(12, 5))
                plot_approach_profile(apch_legs, chart_terrain, ax, db=db)
                plt.tight_layout()
                fig.savefig(chart_path, dpi=150, bbox_inches="tight")
                plt.close(fig)

        if chart_path.exists():
            chart_link = (f'<a class="chart-link" '
                          f'href="charts/{html.escape(chart_filename)}" '
                          f'target="_blank">View</a>')
        else:
            chart_link = ""

        # Clearance styling
        if clearance < 0:
            cls = "clearance-negative"
        elif clearance < m.TERPS_MIN_CLEARANCE_FT:
            cls = "clearance-warning"
        else:
            cls = ""
        clearance_td = (f'<td class="{cls}">{clearance}</td>' if cls
                        else f'<td>{clearance}</td>')

        table_rows.append(
            f"<tr>"
            f"<td>{html.escape(apt)}</td>"
            f"<td>{html.escape(apt_name)}</td>"
            f"<td>{html.escape(state)}</td>"
            f"<td>{html.escape(apch_name)}</td>"
            f"{clearance_td}"
            f"<td>{gpa:.2f}</td>"
            f"<td>{dist:.1f}</td>"
            f"<td>{chart_link}</td>"
            f"</tr>"
        )

    # Build HTML from template
    template_path = Path(__file__).parent / "report_template.html"
    template = template_path.read_text()

    summary = (f"<strong>{n_violations}</strong> approaches with +V advisory "
               f"glidepath clearance &lt; {m.TERPS_MIN_CLEARANCE_FT}&prime;")

    output_html = (template
                   .replace("{{SUMMARY}}", summary)
                   .replace("{{TABLE_ROWS}}", "\n".join(table_rows)))

    index_path = output_dir / "index.html"
    index_path.write_text(output_html)
    n_generated = n_violations - n_cached
    if n_cached:
        print(f"  {n_cached} charts cached, {n_generated} generated")
    print(f"\nHTML report: {index_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Find RNAV (GPS) approaches where +V advisory glidepath "
                    "violates TERPS 250' minimum terrain clearance.")
    parser.add_argument("airport", nargs="?", default=None,
                        help="ICAO airport code (e.g. KSBS); omit for all US")
    parser.add_argument("--html", action="store_true",
                        help="generate HTML report with sortable table and "
                             "profile charts")
    parser.add_argument("--srtm1", action="store_true",
                        help="use high-res SRTM1 (30m) terrain for profile "
                             "charts (default: SRTM3 90m)")
    args = parser.parse_args()

    airport = args.airport.upper() if args.airport else None

    db = m.ApproachDatabase()
    approaches = db.load_approaches(airport=airport)
    if approaches.empty:
        print("No approach geometries extracted.")
        sys.exit(1)

    terrain = db.get_terrain()
    violations = terrain.check_clearance(approaches)

    if args.html:
        generate_html(violations, db, Path("html_output"), srtm1=args.srtm1)
    else:
        print_console(violations)


if __name__ == "__main__":
    main()
