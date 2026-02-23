#!/usr/bin/env python3
"""
Approach Profile View

Draws a profile view for each RNAV (GPS) approach at a given airport, showing:
- Terrain elevation along the approach course
- +V advisory glidepath (or published VNAV glidepath)
- Stepdown altitude constraints
- Waypoint labels

Usage:
    python profile.py KSBS
    python profile.py KOKH
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import approaches as m


def plot_approach_profile(legs, terrain, ax, db=None):
    """Plot a single approach profile on the given axes.

    Args:
        legs: DataFrame of approach legs for one approach (from load_approach_legs)
        terrain: Terrain engine for elevation queries
        ax: matplotlib axes to plot on
        db: ApproachDatabase for MDA lookup (optional)
    """
    # Get approach metadata from the first leg
    apch_name = legs.iloc[0]["apch_name"]
    airport = legs.iloc[0]["airport"]
    threshold_lat = legs.iloc[0]["threshold_lat"]
    threshold_lon = legs.iloc[0]["threshold_lon"]
    threshold_elev = legs.iloc[0]["threshold_elev_ft"]
    tch = legs.iloc[0]["tch_ft"]

    # Find FAF for glidepath computation
    faf = legs[legs["role"] == "FAF"]
    if faf.empty:
        return
    faf = faf.iloc[0]
    faf_alt = faf["altitude1"]
    faf_dist = faf["dist_to_threshold_nm"]

    if np.isnan(faf_alt) or np.isnan(faf_dist) or faf_dist < 0.1:
        return

    # --- Glidepath + terrain profile (shared computation) ---
    profile = terrain.compute_glidepath_profile({
        "faf_lat": faf["lat"], "faf_lon": faf["lon"],
        "faf_alt_ft": faf_alt,
        "threshold_lat": threshold_lat, "threshold_lon": threshold_lon,
        "threshold_elev_ft": threshold_elev, "tch_ft": tch,
    })

    # --- Extended terrain for visualization (IF+0.5NM to threshold) ---
    # The profile covers FAF-to-threshold; we also need terrain context
    # from earlier fixes for the background fill.
    legs_with_coords = legs[legs["lat"].notna()].sort_values(
        "dist_to_threshold_nm", ascending=False)
    start_fix = legs_with_coords.iloc[0]
    fix_dist = start_fix["dist_to_threshold_nm"]
    start_dist = fix_dist + 0.5
    n_vis_samples = max(int(start_dist * m.FEET_PER_NM / m.SAMPLE_INTERVAL_M), 50)
    frac_start = -0.5 / fix_dist
    vis_fracs = np.linspace(frac_start, 1, n_vis_samples + 1)
    vis_lat = start_fix["lat"] + vis_fracs * (threshold_lat - start_fix["lat"])
    vis_lon = start_fix["lon"] + vis_fracs * (threshold_lon - start_fix["lon"])
    vis_dist = m.haversine_nm(vis_lat, vis_lon,
                              np.full_like(vis_lat, threshold_lat),
                              np.full_like(vis_lon, threshold_lon))
    vis_terrain_ft = terrain.get_elevations(vis_lat, vis_lon) * m.FEET_PER_METER

    # --- Plot terrain fill + violation overlay ---
    valid_terrain = vis_terrain_ft[~np.isnan(vis_terrain_ft)]
    terrain_base = max(0, np.min(valid_terrain) - 200) if len(valid_terrain) > 0 else 0
    ax.fill_between(vis_dist, vis_terrain_ft, terrain_base,
                    color="#c4a882", alpha=0.7)
    # Red overlay where +V clearance < 250' (using profile's own sample grid)
    violation = ~np.isnan(profile.clearance_ft) & (
        profile.clearance_ft < m.TERPS_MIN_CLEARANCE_FT)
    ax.fill_between(profile.dist_nm, profile.terrain_ft, terrain_base,
                    where=violation, color="#cc3333", alpha=0.8)
    ax.plot(vis_dist, vis_terrain_ft, color="#8b7355", linewidth=1)

    # --- +V advisory glidepath line ---
    # Check if approach has published VNAV (vert_angle on the FAF leg)
    faf_va = faf["vert_angle"].strip()
    has_vnav = faf_va != "" and faf_va != "000" and faf_va != "0"

    if has_vnav:
        va_str = faf_va
        try:
            va_deg = abs(int(va_str)) / 100.0
        except ValueError:
            va_deg = None

        if va_deg and va_deg > 0:
            vnav_alt = (threshold_elev + tch +
                        profile.dist_nm * m.FEET_PER_NM *
                        np.tan(np.radians(va_deg)))
            ax.plot(profile.dist_nm, vnav_alt, "g-", linewidth=2,
                    label=f"Published VNAV ({va_deg:.2f}\u00b0)")

        # Also show the +V line for comparison
        ax.plot(profile.dist_nm, profile.gp_alt_ft, "r--", linewidth=1.5,
                alpha=0.7,
                label=f"+V advisory ({profile.gpa_deg:.2f}\u00b0)",
                zorder=1)
    else:
        # LNAV-only — +V is the primary guidance
        ax.plot(profile.dist_nm, profile.gp_alt_ft, "r-", linewidth=2,
                label=f"+V advisory ({profile.gpa_deg:.2f}\u00b0)",
                zorder=1)

    # --- Stepdown altitude constraints ---
    # Exclude MAP — its altitude in CIFP is a reference altitude (typically
    # threshold elevation), not the MDA (which CIFP doesn't encode).
    legs_with_alt = legs[
        legs["altitude1"].notna() &
        legs["dist_to_threshold_nm"].notna() &
        (legs["role"] != "MAP")
    ].sort_values("dist_to_threshold_nm", ascending=False)

    color = "#2266cc"
    for i in range(len(legs_with_alt)):
        leg = legs_with_alt.iloc[i]
        alt = leg["altitude1"]
        dist = leg["dist_to_threshold_nm"]

        # Each fix's altitude is the crossing constraint AT that fix —
        # it applies to the segment ending at this fix (from previous fix).
        # After crossing this fix, you may descend to the next constraint.
        prev_dist = (legs_with_alt.iloc[i - 1]["dist_to_threshold_nm"]
                     if i > 0 else start_dist)
        ax.hlines(alt, dist, prev_dist, colors=color, linewidths=1.5,
                  linestyles="-")

        # Altitude label at the start (left end) of the horizontal segment
        ax.text(prev_dist, alt, f"  {int(alt)}'",
                fontsize=7, ha="left", va="bottom", color=color)

        # Vertical drop at this fix to next constraint
        if i + 1 < len(legs_with_alt):
            next_alt = legs_with_alt.iloc[i + 1]["altitude1"]
            ax.vlines(dist, next_alt, alt, colors=color,
                      linewidths=1, linestyles=":")

    # --- MDA line (from OCR'd plate database) ---
    if db is not None:
        mins = db.get_minimums(airport, apch_name)
        # Use LNAV MDA if available; fall back to any circling MDA
        mda = None
        mda_label = "MDA"
        if mins:
            if "LNAV" in mins:
                mda = mins["LNAV"]
            else:
                # Pick first circling-type minimum
                for key, val in mins.items():
                    if "CIRCLING" in key.upper():
                        mda = val
                        mda_label = "MDA (Circling)"
                        break
        if mda is not None and len(legs_with_alt) > 0:
            last_stepdown = legs_with_alt.iloc[-1]
            last_dist = last_stepdown["dist_to_threshold_nm"]
            last_alt = last_stepdown["altitude1"]
            # MAP distance (where MDA segment ends)
            map_leg = legs[legs["role"] == "MAP"]
            map_dist = 0.0
            if not map_leg.empty:
                md = map_leg.iloc[0]["dist_to_threshold_nm"]
                if not np.isnan(md):
                    map_dist = md
            if mda == int(last_alt):
                # MDA equals last stepdown — extend that line to MAP
                ax.hlines(mda, map_dist, last_dist, colors=color,
                          linewidths=1.5, linestyles="-")
                ax.text(last_dist, mda, f"  {mda_label} {mda}'",
                        fontsize=7, ha="left", va="bottom", color=color)
            else:
                # MDA is below last stepdown — draw from last fix to MAP
                ax.vlines(last_dist, mda, last_alt, colors=color,
                          linewidths=1, linestyles=":")
                ax.hlines(mda, map_dist, last_dist, colors=color,
                          linewidths=1.5, linestyles="-")
                ax.text(last_dist, mda, f"  {mda_label} {mda}'",
                        fontsize=7, ha="left", va="bottom", color=color)

    # --- Waypoint labels ---
    for _, leg in legs.iterrows():
        dist = leg["dist_to_threshold_nm"]
        fix_id = leg["fix_id"]
        if np.isnan(dist) or not fix_id:
            continue

        alt = leg["altitude1"]
        label_y = alt if not np.isnan(alt) else threshold_elev
        role_label = f" ({leg['role']})" if leg["role"] else ""
        ax.annotate(
            f"{fix_id}{role_label}",
            xy=(dist, label_y),
            xytext=(0, 20), textcoords="offset points",
            fontsize=7, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="green", alpha=0.8),
            arrowprops=dict(arrowstyle="-", color="green", lw=1),
        )

    # --- Formatting ---
    ax.set_title(f"{airport}  {apch_name}", fontweight="bold")
    ax.set_xlabel("Distance from threshold (NM)")
    ax.set_ylabel("Altitude (ft MSL)")
    ax.invert_xaxis()  # FAF on left, threshold on right
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Set y-axis to show terrain, glidepath, and all constraints
    constraint_alts = legs_with_alt["altitude1"].values
    all_alts = np.concatenate([
        vis_terrain_ft[~np.isnan(vis_terrain_ft)],
        profile.gp_alt_ft, constraint_alts])
    y_min = max(0, np.nanmin(all_alts) - 300)
    y_max = np.nanmax(all_alts) + 500
    ax.set_ylim(y_min, y_max)


def main():
    if len(sys.argv) < 2:
        print("Usage: python profile.py <AIRPORT_ICAO>")
        print("Example: python profile.py KSBS")
        sys.exit(1)

    airport = sys.argv[1].upper()
    print(f"Loading approach data for {airport}...")

    db = m.ApproachDatabase()
    legs = db.load_approach_legs(airport=airport)
    if legs.empty:
        print(f"No RNAV (GPS) approaches found for {airport}")
        sys.exit(1)

    terrain = db.get_terrain(srtm1=True)

    # One subplot per approach
    approaches = legs.groupby(["airport", "proc_id"])
    n_approaches = len(approaches)
    print(f"Plotting {n_approaches} approach(es)...")

    fig, axes = plt.subplots(n_approaches, 1,
                             figsize=(12, 5 * n_approaches),
                             squeeze=False)

    for i, ((apt, pid), apch_legs) in enumerate(approaches):
        plot_approach_profile(apch_legs, terrain, axes[i, 0], db=db)

    plt.tight_layout()
    out_path = f"{airport}_profile.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
