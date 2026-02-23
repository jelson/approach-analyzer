#!/usr/bin/env python3
"""
Find all US RNAV (GPS) approaches where the +V advisory glidepath
violates the TERPS 250' minimum terrain clearance.

Usage:
    python approach_study.py              # all US approaches
    python approach_study.py KSBS         # single airport
"""

import sys

import approaches as m


def main():
    airport = sys.argv[1].upper() if len(sys.argv) > 1 else None

    db = m.ApproachDatabase()
    approaches = db.load_approaches(airport=airport)
    if approaches.empty:
        print("No approach geometries extracted.")
        sys.exit(1)

    terrain = db.get_terrain()
    violations = terrain.check_clearance(approaches)

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


if __name__ == "__main__":
    main()
