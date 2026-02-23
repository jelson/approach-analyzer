#!/usr/bin/env python3
"""Top 10 most dangerous +V approaches in Washington and Oregon."""

import approaches as m

# WA+OR bounding box (generous)
MIN_LAT, MAX_LAT = 42.0, 49.0
MIN_LON, MAX_LON = -124.8, -116.5

db = m.ApproachDatabase()
approaches = db.load_approaches()
wa_or = approaches[
    approaches["threshold_lat"].between(MIN_LAT, MAX_LAT) &
    approaches["threshold_lon"].between(MIN_LON, MAX_LON)
].reset_index(drop=True)
print(f"\nWA/OR approaches: {len(wa_or)}")

terrain = db.get_terrain()
results = terrain.check_clearance(wa_or, min_clearance_ft=99999)
top10 = results.head(10)

print("\n" + "=" * 75)
print("TOP 10 MOST DANGEROUS +V APPROACHES IN WA & OR")
print("=" * 75)
for i, row in enumerate(top10.itertuples(), 1):
    marker = " ***" if row.worst_clearance_ft < 250 else ""
    print(f"  {i:>2}. {row.airport} {row.apch_name:<30} "
          f"{row.worst_clearance_ft:>5}'  "
          f"GPA: {row.gpa_deg}  dist: {row.distance_nm} NM"
          f"{marker}")

n_viol = (top10["worst_clearance_ft"] < 250).sum()
print(f"\n{n_viol} of {len(top10)} violate the 250' TERPS minimum.")
