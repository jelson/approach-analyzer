#!/usr/bin/env python3
"""Top 10 most dangerous +V approaches in Washington and Oregon."""

import approaches as m

# WA+OR bounding box (generous)
MIN_LAT, MAX_LAT = 42.0, 49.0
MIN_LON, MAX_LON = -124.8, -116.5

db = m.ApproachDatabase()
approaches = db.get_approaches()
wa_or = approaches[
    approaches["threshold_lat"].between(MIN_LAT, MAX_LAT) &
    approaches["threshold_lon"].between(MIN_LON, MAX_LON)
].reset_index(drop=True)
print(f"\nWA/OR approaches: {len(wa_or)}")

terrain = db.get_terrain()
analyses = terrain.check_clearance(wa_or, db)

# Sort by worst clearance, take top 10
with_clearance = [a for a in analyses if a.worst_clearance_ft is not None]
with_clearance.sort(key=lambda a: a.worst_clearance_ft or 0)
top10 = with_clearance[:10]

print("\n" + "=" * 75)
print("TOP 10 MOST DANGEROUS +V APPROACHES IN WA & OR")
print("=" * 75)
for i, a in enumerate(top10, 1):
    assert a.worst_clearance_ft is not None
    marker = " ***" if a.worst_clearance_ft < 250 else ""
    print(f"  {i:>2}. {a.airport} {a.apch_name:<30} "
          f"{a.worst_clearance_ft:>5}'  "
          f"GPA: {a.gpa_deg}  dist: {a.profile.distance_nm:.1f} NM"
          f"{marker}")

n_viol = sum(1 for a in top10
             if a.worst_clearance_ft is not None and a.worst_clearance_ft < 250)
print(f"\n{n_viol} of {len(top10)} violate the 250' TERPS minimum.")
