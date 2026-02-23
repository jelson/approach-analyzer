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


def plot_approach_profile(legs, terrain, ax, db=None,
                          worst_dist_nm=None, worst_terrain_ft=None):
    """Plot a single approach profile on the given axes.

    Args:
        legs: DataFrame of approach legs for one approach (from get_approach_legs)
        terrain: Terrain engine for elevation queries
        ax: matplotlib axes to plot on
        db: ApproachDatabase for MDA lookup (optional)
        worst_dist_nm: pre-computed worst +V clearance distance from threshold;
            if provided (with worst_terrain_ft), the annotation uses these
            values instead of recomputing from the profile's own terrain samples
        worst_terrain_ft: pre-computed terrain elevation at the worst point
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

    # --- Extended terrain for visualization ---
    # Before the FAF: sample along the IF-to-FAF line for context terrain.
    # From FAF to threshold: use the profile's own samples so the brown
    # background and red violation overlay follow the same ground track.
    legs_with_coords = legs[legs["lat"].notna()].sort_values(
        "dist_to_threshold_nm", ascending=False)
    start_fix = legs_with_coords.iloc[0]
    fix_dist = start_fix["dist_to_threshold_nm"]
    start_dist = fix_dist + 0.5

    # Pre-FAF segment: IF(+0.5NM) to FAF along the IF-FAF line
    pre_faf_dist_range = start_dist - faf_dist
    n_pre = max(int(pre_faf_dist_range * m.FEET_PER_NM / m.SAMPLE_INTERVAL_M), 10)
    pre_fracs = np.linspace(-0.5 / fix_dist,
                            1.0 - faf_dist / fix_dist, n_pre)
    pre_lat = start_fix["lat"] + pre_fracs * (threshold_lat - start_fix["lat"])
    pre_lon = start_fix["lon"] + pre_fracs * (threshold_lon - start_fix["lon"])
    pre_dist = m.haversine_nm(pre_lat, pre_lon,
                              np.full_like(pre_lat, threshold_lat),
                              np.full_like(pre_lon, threshold_lon))
    pre_terrain = terrain.get_elevations(pre_lat, pre_lon) * m.FEET_PER_METER

    # Combine pre-FAF terrain with profile's FAF-to-threshold terrain
    vis_dist = np.concatenate([pre_dist, profile.dist_nm])
    vis_terrain_ft = np.concatenate([pre_terrain, profile.terrain_ft])

    # --- Plot terrain fill ---
    # Single terrain array; color red where clearance < 250', brown elsewhere.
    # Pre-FAF samples are always brown (no clearance data there).
    valid_terrain = vis_terrain_ft[~np.isnan(vis_terrain_ft)]
    terrain_base = max(0, np.min(valid_terrain) - 200) if len(valid_terrain) > 0 else 0
    violation = np.concatenate([
        np.zeros(len(pre_dist), dtype=bool),
        ~np.isnan(profile.clearance_ft) & (
            profile.clearance_ft < m.TERPS_MIN_CLEARANCE_FT),
    ])
    ax.fill_between(vis_dist, vis_terrain_ft, terrain_base,
                    where=~violation, color="#c4a882", alpha=0.7)
    ax.fill_between(vis_dist, vis_terrain_ft, terrain_base,
                    where=violation, color="#cc3333", alpha=0.8,
                    label=f"+V clearance < {m.TERPS_MIN_CLEARANCE_FT}'")
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

    # --- Stepdown altitude constraints + MDA (from shared staircase) ---
    staircase = m.build_lnav_staircase(legs, db=db, start_dist=start_dist)

    color = "#2266cc"
    for i, (from_dist, to_dist, alt) in enumerate(staircase.segments):
        ax.hlines(alt, to_dist, from_dist, colors=color, linewidths=1.5,
                  linestyles="-")
        ax.text(from_dist, alt, f"  {int(alt)}'",
                fontsize=7, ha="left", va="bottom", color=color)

        # Vertical drop to next segment
        if i + 1 < len(staircase.segments):
            next_alt = staircase.segments[i + 1][2]
            ax.vlines(to_dist, next_alt, alt, colors=color,
                      linewidths=1, linestyles=":")

    # MDA line (extends from last stepdown to MAP)
    if staircase.mda is not None and staircase.segments:
        last_dist = staircase.segments[-1][1]
        last_alt = staircase.segments[-1][2]
        if staircase.mda == int(last_alt):
            # MDA equals last stepdown — extend that line to MAP
            ax.hlines(staircase.mda, staircase.map_dist, last_dist,
                      colors=color, linewidths=1.5, linestyles="-")
            ax.text(last_dist, staircase.mda,
                    f"  {staircase.mda_label} {staircase.mda}'",
                    fontsize=7, ha="left", va="bottom", color=color)
        else:
            # MDA is below last stepdown — draw from last fix to MAP
            ax.vlines(last_dist, staircase.mda, last_alt, colors=color,
                      linewidths=1, linestyles=":")
            ax.hlines(staircase.mda, staircase.map_dist, last_dist,
                      colors=color, linewidths=1.5, linestyles="-")
            ax.text(last_dist, staircase.mda,
                    f"  {staircase.mda_label} {staircase.mda}'",
                    fontsize=7, ha="left", va="bottom", color=color)

    # --- LNAV clearance annotation at worst +V clearance point ---
    # Use pre-computed worst point if provided, otherwise find it from profile
    if worst_dist_nm is not None and worst_terrain_ft is not None:
        worst_dist = worst_dist_nm
        worst_terrain = worst_terrain_ft
    else:
        valid_clearance = ~np.isnan(profile.clearance_ft)
        if valid_clearance.any():
            worst_i = np.nanargmin(profile.clearance_ft)
            worst_dist = profile.dist_nm[worst_i]
            worst_terrain = profile.terrain_ft[worst_i]
        else:
            worst_dist = None
            worst_terrain = None

    if worst_dist is not None and worst_terrain is not None and staircase.segments:
        lnav_alt = staircase.altitude_at(worst_dist)

        if lnav_alt is not None and not np.isnan(worst_terrain):
            lnav_clearance = int(round(lnav_alt - worst_terrain))
            ann_color = "#cc3333" if lnav_clearance < m.TERPS_MIN_CLEARANCE_FT \
                else "#6a0dad"
            # Vertical line from terrain to LNAV altitude
            ax.vlines(worst_dist, worst_terrain, lnav_alt,
                      colors=ann_color, linewidths=2, linestyles="-",
                      zorder=5)
            # Label above the line, styled like waypoint labels
            ax.annotate(
                f"Proc Clearance\n{lnav_clearance}'",
                xy=(worst_dist, lnav_alt),
                xytext=(0, 16), textcoords="offset points",
                fontsize=7, ha="center", va="bottom",
                color=ann_color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=ann_color, alpha=0.8),
                zorder=5,
            )

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
    constraint_alts = np.array([alt for _, _, alt in staircase.segments])
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
    legs = db.get_approach_legs(airport=airport)
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
