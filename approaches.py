"""
Approach terrain clearance library.

Provides access to FAA RNAV (GPS) approach data from two sources:
  - CIFP (Coded Instrument Flight Procedures) in ARINC 424 format
  - OCR'd approach plate minimums (ammaraskar/faa-instrument-approach-db)

Also provides SRTM terrain elevation engines for clearance analysis.

This module has no main(); see approach_study.py, wa_or_study.py, profile.py.
"""

import csv
import io
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path

import arinc424
import numpy as np
import pandas as pd
import requests


# =============================================================================
# Constants
# =============================================================================

TERPS_MIN_CLEARANCE_FT = 250  # TERPS 8-1-4 final approach area minimum
FEET_PER_METER = 3.28084
EARTH_RADIUS_NM = 3440.065    # nautical miles
FEET_PER_NM = 6076.12
NM_PER_METER = 1.0 / 1852.0
SAMPLE_INTERVAL_M = 30        # terrain sample interval in meters
DEFAULT_TCH_FT = 50           # default threshold crossing height when CIFP omits it
# On flat terrain with a 3° glidepath, +V is at 250' AGL at 0.785 NM from
# threshold (250 / tan(3°) / 6076 ft/NM), so closer than ~0.8 NM can't
# violate the 250' TERPS minimum regardless of terrain.
MIN_CLEARANCE_DIST_NM = 0.8

SRTM3_SAMPLES = 1201  # 3-arc-second SRTM tile is 1201x1201
SRTM3_URL_BASE = "https://srtm.kurviger.de/SRTM3/"


# =============================================================================
# Glidepath Profile Types
# =============================================================================

@dataclass
class GlidepathProfile:
    """Glidepath and terrain sampled along the FAF-to-threshold segment.

    All distances are in NM from the runway threshold.
    All altitudes/elevations are in feet MSL.
    """
    dist_nm: np.ndarray       # distance from threshold at each sample (NM)
    gp_alt_ft: np.ndarray     # +V glidepath altitude (ft MSL)
    terrain_ft: np.ndarray    # terrain elevation (ft MSL; NaN if unavailable)
    clearance_ft: np.ndarray  # gp_alt_ft - terrain_ft
    lat: np.ndarray           # latitude of each sample point
    lon: np.ndarray           # longitude of each sample point
    gpa_deg: float            # computed glidepath angle (degrees)
    distance_nm: float        # FAF-to-threshold distance (NM)

    def worst_clearance_point(self) -> tuple[float, float] | None:
        """Find the point of worst +V terrain clearance.

        Only considers points beyond MIN_CLEARANCE_DIST_NM from the threshold.
        Returns (dist_nm, terrain_ft) or None if no valid clearance data.
        """
        valid = (~np.isnan(self.clearance_ft)
                 & (self.dist_nm >= MIN_CLEARANCE_DIST_NM))
        if not valid.any():
            return None
        # Mask invalid points so argmin ignores them
        masked = np.where(valid, self.clearance_ft, np.inf)
        worst_i = np.argmin(masked)
        return float(self.dist_nm[worst_i]), float(self.terrain_ft[worst_i])


@dataclass
class _SampleResult:
    """Vectorized glidepath + terrain samples for multiple approaches."""
    approach_idx: np.ndarray   # which approach each sample belongs to
    dist_nm: np.ndarray        # distance from threshold (NM) per sample
    gp_alt_ft: np.ndarray      # glidepath altitude (ft MSL) per sample
    terrain_ft: np.ndarray     # terrain elevation (ft MSL) per sample
    clearance_ft: np.ndarray   # gp_alt - terrain per sample
    lat: np.ndarray            # sample latitude
    lon: np.ndarray            # sample longitude
    gpa_deg: np.ndarray        # GPA per approach (length = n_approaches)
    distance_nm: np.ndarray    # FAF-to-threshold dist per approach


@dataclass
class LnavStaircase:
    """LNAV dive-and-drive altitude staircase for one approach.

    Represents the minimum altitude profile a pilot should follow on an LNAV
    approach: each segment's altitude is the crossing constraint at the next
    fix (the fix the pilot is flying toward).

    segments: list of (from_dist, to_dist, altitude) tuples in NM from
              threshold, sorted from FAF toward threshold.
    mda: MDA altitude in feet MSL, or None if unavailable.
    mda_label: label for the MDA (e.g. "MDA" or "MDA (Circling)").
    map_dist: MAP distance from threshold in NM.
    """
    segments: list[tuple[float, float, float]]
    mda: float | None
    mda_label: str
    map_dist: float

    def altitude_at(self, dist_nm: float) -> float | None:
        """Return the LNAV minimum altitude at a given distance from threshold.

        Walks the staircase from FAF toward threshold. In the segment after
        the last fix, returns MDA. Returns None if no altitude data.
        """
        for from_dist, to_dist, alt in self.segments:
            if dist_nm >= to_dist:
                return alt
        # Past all segments — use MDA or last segment altitude
        if self.mda is not None:
            return self.mda
        if self.segments:
            return self.segments[-1][2]
        return None


@dataclass
class ApproachAnalysis:
    """Complete analysis of one approach: glidepath profile, staircase, clearances.

    Produced by Terrain.check_clearance() — the single entry point for all
    terrain clearance computations. Used by both the HTML table/console output
    and the profile chart, ensuring consistent values everywhere.
    """
    airport: str
    proc_id: str
    apch_name: str
    profile: GlidepathProfile
    staircase: LnavStaircase
    gpa_deg: float
    worst_clearance_ft: int | None
    worst_dist_nm: float | None
    worst_terrain_ft: float | None
    lnav_clearance_ft: int | None


def build_lnav_staircase(legs: pd.DataFrame,
                         db: "ApproachDatabase | None" = None,
                         start_dist: float | None = None,
                         ) -> LnavStaircase:
    """Build the LNAV altitude staircase from approach legs.

    This is the single source of truth for the dive-and-drive staircase used
    for both chart drawing and LNAV clearance computation.

    Args:
        legs: DataFrame of approach legs for one approach (from get_approach_legs)
        db: ApproachDatabase for MDA lookup (optional)
        start_dist: distance from threshold where the staircase drawing starts
                    (typically IF distance + 0.5 NM). If None, uses the
                    farthest fix distance.
    """
    airport = legs.iloc[0]["airport"]
    apch_name = legs.iloc[0]["apch_name"]

    # Legs with altitude constraints, sorted FAF-first (descending distance)
    legs_with_alt = legs[
        legs["altitude1"].notna() &
        legs["dist_to_threshold_nm"].notna() &
        (legs["role"] != "MAP")
    ].sort_values("dist_to_threshold_nm", ascending=False)

    if start_dist is None and len(legs_with_alt) > 0:
        start_dist = legs_with_alt.iloc[0]["dist_to_threshold_nm"]

    # Build segments: each fix's altitude applies from the previous fix to
    # this fix (the segment the pilot traverses while descending to this
    # fix's constraint).
    segments: list[tuple[float, float, float]] = []
    for i in range(len(legs_with_alt)):
        alt: float = legs_with_alt.iloc[i]["altitude1"]
        dist: float = legs_with_alt.iloc[i]["dist_to_threshold_nm"]
        prev: float = (legs_with_alt.iloc[i - 1]["dist_to_threshold_nm"]
                       if i > 0 else start_dist or dist)
        segments.append((prev, dist, alt))

    # MDA from plate OCR database
    mda = None
    mda_label = "MDA"
    if db is not None:
        mins = db.get_minimums(airport, apch_name)
        if mins:
            if "LNAV" in mins:
                mda = mins["LNAV"]
            else:
                for key, val in mins.items():
                    if "CIRCLING" in key.upper():
                        mda = val
                        mda_label = "MDA (Circling)"
                        break

    # MAP distance
    map_leg = legs[legs["role"] == "MAP"]
    map_dist = 0.0
    if not map_leg.empty:
        md = map_leg.iloc[0]["dist_to_threshold_nm"]
        if not np.isnan(md):
            map_dist = md

    return LnavStaircase(
        segments=segments, mda=mda, mda_label=mda_label, map_dist=map_dist)


# =============================================================================
# CIFP Download
# =============================================================================

def download_cifp(data_dir: Path) -> Path:
    """Download the current CIFP from the FAA website."""
    cifp_path = data_dir / "FAACIFP18"
    if cifp_path.exists():
        print(f"CIFP already exists: {cifp_path}")
        return cifp_path

    print("Finding current CIFP download URL...")
    import re
    r = requests.get(
        "https://www.faa.gov/air_traffic/flight_info/aeronav/digital_products/cifp/download/",
        timeout=30,
    )
    r.raise_for_status()
    links = re.findall(r'href=["\']([^"\']*CIFP[^"\']*\.zip)["\']', r.text, re.IGNORECASE)
    if not links:
        raise RuntimeError("Could not find CIFP download link on FAA website")

    url = links[0]
    print(f"Downloading CIFP from {url}...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    print(f"  Downloaded {len(r.content) / 1024 / 1024:.1f} MB")

    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(data_dir)
    print(f"  Extracted to {data_dir}")
    return cifp_path


# =============================================================================
# SRTM Terrain Data
# =============================================================================

def srtm_tile_name(lat: float, lon: float) -> str:
    """Return the SRTM tile filename for a given lat/lon (e.g., 'N40W107')."""
    lat_prefix = "N" if lat >= 0 else "S"
    lon_prefix = "E" if lon >= 0 else "W"
    return f"{lat_prefix}{abs(int(math.floor(lat))):02d}{lon_prefix}{abs(int(math.floor(lon))):03d}"


def load_srtm_tile_index() -> dict[str, str]:
    """Load the SRTM tile URL index from the srtm.py package's list.json."""
    import srtm as srtm_pkg
    list_path = Path(srtm_pkg.__file__).parent / "list.json"
    with open(list_path) as f:
        data = json.load(f)
    # The JSON has keys "srtm1" and "srtm3", each mapping "NxxWyyy.hgt" -> URL
    index = {}
    srtm3 = data.get("srtm3", {})
    for name, url in srtm3.items():
        tile = name.replace(".hgt", "")
        index[tile] = url
    print(f"  Loaded SRTM3 tile index: {len(index)} tiles available")
    return index


class Terrain:
    """Base class for terrain elevation engines."""

    def get_elevation(self, lat: float, lon: float) -> float | None:
        raise NotImplementedError

    def get_elevations(self, lats: np.ndarray,
                       lons: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _sample_glidepaths(self, approaches: pd.DataFrame,
                           min_dist_nm: float = 0.0) -> _SampleResult:
        """Vectorized glidepath + terrain sampling for multiple approaches.

        Computes GPA and FAF-to-threshold distance from raw approach facts.

        Args:
            approaches: DataFrame with columns faf_lat, faf_lon, faf_alt_ft,
                threshold_lat, threshold_lon, threshold_elev_ft, tch_ft
            min_dist_nm: stop sampling this far from threshold (NM)

        Returns _SampleResult with flat sample arrays and per-approach values.
        """
        # Compute FAF-to-threshold distance and GPA from raw facts
        dist_nm = haversine_nm(
            approaches["faf_lat"].values, approaches["faf_lon"].values,
            approaches["threshold_lat"].values,
            approaches["threshold_lon"].values,
        )
        alt_drop = (approaches["faf_alt_ft"].values
                    - approaches["threshold_elev_ft"].values
                    - approaches["tch_ft"].values)
        gpa_rad = np.arctan2(alt_drop, dist_nm * FEET_PER_NM)
        gpa_deg = np.degrees(gpa_rad)

        # Generate sample points along each approach
        sample_interval_nm = SAMPLE_INTERVAL_M * NM_PER_METER
        check_dist = dist_nm - min_dist_nm
        n_samples = np.maximum(
            (check_dist / sample_interval_nm).astype(int), 2)

        # Build flat arrays for vectorized sampling across all approaches.
        # Each approach gets its own linspace of fractional positions (0=FAF,
        # 1=threshold). approach_indices tracks which approach each sample
        # belongs to; after concatenation, these flat arrays let all the
        # downstream math run as single vectorized operations.
        n_approaches = len(approaches)
        approach_indices = []
        fracs_all = []
        for i in range(n_approaches):
            if check_dist[i] <= 0 or dist_nm[i] <= 0:
                continue
            ns = n_samples[i]
            max_frac = check_dist[i] / dist_nm[i]
            fracs = np.linspace(0, max_frac, ns + 1)
            approach_indices.append(np.full(ns + 1, i))
            fracs_all.append(fracs)

        approach_idx = np.concatenate(approach_indices)
        fracs = np.concatenate(fracs_all)

        # Expand per-approach values to per-sample arrays: approach_idx
        # maps each sample to its approach, so fancy-indexing repeats each
        # approach's value for all of that approach's samples.
        per_sample_faf_lat = approaches["faf_lat"].values[approach_idx]
        per_sample_faf_lon = approaches["faf_lon"].values[approach_idx]
        per_sample_thr_lat = approaches["threshold_lat"].values[approach_idx]
        per_sample_thr_lon = approaches["threshold_lon"].values[approach_idx]

        # Linearly interpolate lat/lon along the FAF-to-threshold segment
        # at each sample fraction (flat-earth approx, fine for ~5-10 NM)
        per_sample_lat = per_sample_faf_lat + fracs * (per_sample_thr_lat - per_sample_faf_lat)
        per_sample_lon = per_sample_faf_lon + fracs * (per_sample_thr_lon - per_sample_faf_lon)

        # Haversine distance from each sample to its threshold
        per_sample_dist = haversine_nm(per_sample_lat, per_sample_lon,
                                       per_sample_thr_lat, per_sample_thr_lon)

        # Glidepath altitude at each sample
        per_sample_thr_elev = approaches["threshold_elev_ft"].values[approach_idx]
        per_sample_tch = approaches["tch_ft"].values[approach_idx]
        per_sample_gpa = gpa_rad[approach_idx]
        per_sample_gp_alt = (per_sample_thr_elev + per_sample_tch +
                             per_sample_dist * FEET_PER_NM * np.tan(per_sample_gpa))

        # Terrain query
        per_sample_terrain = self.get_elevations(per_sample_lat, per_sample_lon) * FEET_PER_METER
        per_sample_clearance = per_sample_gp_alt - per_sample_terrain

        return _SampleResult(
            approach_idx=approach_idx,
            dist_nm=per_sample_dist,
            gp_alt_ft=per_sample_gp_alt,
            terrain_ft=per_sample_terrain,
            clearance_ft=per_sample_clearance,
            lat=per_sample_lat,
            lon=per_sample_lon,
            gpa_deg=gpa_deg,
            distance_nm=dist_nm,
        )

    def check_clearance(self, approaches: pd.DataFrame,
                        db: "ApproachDatabase",
                        ) -> list[ApproachAnalysis]:
        """Analyze terrain clearance for each approach.

        This is the single entry point for all terrain clearance computations.
        Returns one ApproachAnalysis per approach with glidepath profile,
        LNAV staircase, and clearance metadata.  Callers filter the result
        to find violations.

        Args:
            approaches: geometry DataFrame from get_approaches()
            db: ApproachDatabase for legs and MDA lookup
        """
        print("\nChecking terrain clearance...")
        print(f"  {len(approaches)} approaches to check")

        apch = approaches.copy().reset_index(drop=True)
        if apch.empty:
            print("  No approaches to check.")
            return []

        # Load approach legs (uses CIFP cache from get_approaches())
        airports = apch["airport"].unique()
        airport_filter = airports[0] if len(airports) == 1 else None
        legs = db.get_approach_legs(airport=airport_filter)

        # Batch-vectorized terrain query for all approaches at once
        print("  Querying terrain elevations...")
        samples = self._sample_glidepaths(apch)
        print(f"  Generated {len(samples.dist_nm):,} sample points")

        # Split batch result into per-approach ApproachAnalysis objects
        results: list[ApproachAnalysis] = []
        n_approaches = len(apch)

        for i in range(n_approaches):
            mask = samples.approach_idx == i
            if not mask.any():
                continue

            airport = apch.loc[i, "airport"]
            proc_id = apch.loc[i, "proc_id"]
            apch_name = apch.loc[i, "apch_name"]

            profile = GlidepathProfile(
                dist_nm=samples.dist_nm[mask],
                gp_alt_ft=samples.gp_alt_ft[mask],
                terrain_ft=samples.terrain_ft[mask],
                clearance_ft=samples.clearance_ft[mask],
                lat=samples.lat[mask],
                lon=samples.lon[mask],
                gpa_deg=float(samples.gpa_deg[i]),
                distance_nm=float(samples.distance_nm[i]),
            )

            # Build LNAV staircase from approach legs
            apch_legs = legs[
                (legs["airport"] == airport) & (legs["proc_id"] == proc_id)]
            staircase = build_lnav_staircase(apch_legs, db=db)

            # Find worst +V clearance point and compute proc clearance
            worst_point = profile.worst_clearance_point()
            if worst_point is not None:
                worst_dist, worst_terrain = worst_point
                gp_alt_at_worst = float(np.interp(
                    worst_dist, profile.dist_nm[::-1],
                    profile.gp_alt_ft[::-1]))
                worst_clearance = int(round(gp_alt_at_worst - worst_terrain))
                lnav_alt = staircase.altitude_at(worst_dist)
                lnav_clearance = (int(round(lnav_alt - worst_terrain))
                                  if lnav_alt is not None else None)
            else:
                worst_clearance = None
                worst_dist = None
                worst_terrain = None
                lnav_clearance = None

            results.append(ApproachAnalysis(
                airport=airport,
                proc_id=proc_id,
                apch_name=apch_name,
                profile=profile,
                staircase=staircase,
                gpa_deg=round(float(samples.gpa_deg[i]), 2),
                worst_clearance_ft=worst_clearance,
                worst_dist_nm=round(worst_dist, 3) if worst_dist else None,
                worst_terrain_ft=(int(round(worst_terrain))
                                  if worst_terrain is not None else None),
                lnav_clearance_ft=lnav_clearance,
            ))

        n_violations = sum(1 for a in results
                           if a.worst_clearance_ft is not None
                           and a.worst_clearance_ft < TERPS_MIN_CLEARANCE_FT)
        print(f"  Found {n_violations} approaches violating "
              f"{TERPS_MIN_CLEARANCE_FT}' clearance")
        return results


class SRTMTerrain(Terrain):
    """SRTM terrain elevation queries with local caching."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_tiles: dict[str, np.ndarray | None] = {}
        self.tile_index: dict[str, str] | None = None

    def _ensure_index(self) -> None:
        if self.tile_index is None:
            self.tile_index = load_srtm_tile_index()

    def _load_tile(self, tile_name: str) -> np.ndarray | None:
        """Load an SRTM tile, downloading if necessary. Returns 1201x1201 numpy array."""
        if tile_name in self.loaded_tiles:
            return self.loaded_tiles[tile_name]

        hgt_path = self.cache_dir / f"{tile_name}.hgt"

        if not hgt_path.exists():
            self._download_tile(tile_name, hgt_path)

        if not hgt_path.exists():
            self.loaded_tiles[tile_name] = None
            return None

        # Read the .hgt file: 1201x1201 big-endian signed 16-bit integers
        data = np.fromfile(hgt_path, dtype=">i2")
        expected = SRTM3_SAMPLES * SRTM3_SAMPLES
        if len(data) != expected:
            # Might be SRTM1 (3601x3601) - resample to SRTM3
            srtm1_samples = 3601
            if len(data) == srtm1_samples * srtm1_samples:
                data = data.reshape((srtm1_samples, srtm1_samples))
                # Subsample to ~1201 points
                step = 3
                data = data[::step, ::step][:SRTM3_SAMPLES, :SRTM3_SAMPLES]
            else:
                print(f"  Warning: unexpected tile size for {tile_name}: {len(data)} samples")
                return None

        data = data.reshape((SRTM3_SAMPLES, SRTM3_SAMPLES))
        self.loaded_tiles[tile_name] = data
        return data

    def _download_tile(self, tile_name: str, hgt_path: Path) -> None:
        """Download an SRTM tile from the mirror."""
        self._ensure_index()
        assert self.tile_index is not None
        url = self.tile_index.get(tile_name)
        if not url:
            print(f"  Warning: no SRTM data available for tile {tile_name}")
            return

        print(f"  Downloading SRTM tile {tile_name}...")
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  Warning: failed to download {tile_name}: {e}")
            return

        # The download is a .hgt.zip file
        if url.endswith(".zip"):
            z = zipfile.ZipFile(io.BytesIO(r.content))
            for name in z.namelist():
                if name.endswith(".hgt"):
                    z.extract(name, self.cache_dir)
                    extracted = self.cache_dir / name
                    if extracted != hgt_path:
                        extracted.rename(hgt_path)
                    break
        else:
            hgt_path.write_bytes(r.content)

    def get_elevation(self, lat: float, lon: float) -> float | None:
        """Get terrain elevation in meters at a given lat/lon. Returns None if unavailable."""
        result = self.get_elevations(np.array([lat]), np.array([lon]))
        return None if np.isnan(result[0]) else float(result[0])

    def get_elevations(self, lats: np.ndarray,
                       lons: np.ndarray) -> np.ndarray:
        """Batch terrain elevation query using vectorized numpy operations."""
        lats = np.asarray(lats, dtype=float)
        lons = np.asarray(lons, dtype=float)
        result = np.full(len(lats), np.nan)

        # Group points by tile using floor of coordinates
        tile_lats = np.floor(lats).astype(int)
        tile_lons = np.floor(lons).astype(int)

        # Get unique tiles
        tile_pairs = np.column_stack([tile_lats, tile_lons])
        unique_pairs = np.unique(tile_pairs, axis=0)

        for tlat, tlon in unique_pairs:
            # Find points in this tile
            mask = (tile_lats == tlat) & (tile_lons == tlon)

            # Load tile (uses any point in the tile to compute tile name)
            tile_name = srtm_tile_name(float(tlat) + 0.5, float(tlon) + 0.5)
            tile = self._load_tile(tile_name)
            if tile is None:
                continue

            # Vectorized bilinear interpolation for all points in this tile
            t_lats = lats[mask]
            t_lons = lons[mask]

            frac_lat = t_lats - tlat
            frac_lon = t_lons - tlon

            row_f = (1.0 - frac_lat) * (SRTM3_SAMPLES - 1)
            col_f = frac_lon * (SRTM3_SAMPLES - 1)

            row0 = np.floor(row_f).astype(int)
            col0 = np.floor(col_f).astype(int)
            row1 = np.minimum(row0 + 1, SRTM3_SAMPLES - 1)
            col1 = np.minimum(col0 + 1, SRTM3_SAMPLES - 1)

            dr = row_f - row0
            dc = col_f - col0

            z00 = tile[row0, col0].astype(float)
            z01 = tile[row0, col1].astype(float)
            z10 = tile[row1, col0].astype(float)
            z11 = tile[row1, col1].astype(float)

            elev = (z00 * (1 - dr) * (1 - dc) +
                    z01 * (1 - dr) * dc +
                    z10 * dr * (1 - dc) +
                    z11 * dr * dc)

            # Handle SRTM void values (-32768)
            void_mask = (z00 == -32768) | (z01 == -32768) | (z10 == -32768) | (z11 == -32768)
            elev[void_mask] = np.nan

            result[mask] = elev

        return result


class SRTMLibTerrain(Terrain):
    """Terrain queries using the srtm.py library (supports SRTM1 30m resolution).

    Uses the library's own tile downloading and caching. Slower than
    SRTMTerrain for large batch queries, but supports the highest resolution
    SRTM1 data and is convenient for single-airport analysis.
    """

    def __init__(self, srtm1: bool = True) -> None:
        import srtm as srtm_pkg
        print(f"  Initializing srtm.py library (srtm1={srtm1})...")
        self.data = srtm_pkg.get_data(srtm1=srtm1, srtm3=True)

    def get_elevation(self, lat: float, lon: float) -> float | None:
        """Get terrain elevation in meters at a given lat/lon."""
        return self.data.get_elevation(lat, lon)

    def get_elevations(self, lats: np.ndarray,
                       lons: np.ndarray) -> np.ndarray:
        """Batch terrain query. Per-point via srtm library."""
        lats = np.asarray(lats, dtype=float)
        lons = np.asarray(lons, dtype=float)
        result = np.full(len(lats), np.nan)
        for i in range(len(lats)):
            elev = self.data.get_elevation(float(lats[i]), float(lons[i]))
            if elev is not None:
                result[i] = elev
        return result


# =============================================================================
# ARINC 424 / CIFP Parsing
# =============================================================================

def parse_arinc_latlon(s: str) -> float | None:
    """Parse ARINC 424 lat/lon like 'N40304424' or 'W106514909'.

    Format: H DD MM SS.ss (hemisphere, degrees, minutes, seconds*100)
    """
    if not s or len(s) < 9 or s.strip() == "":
        return None

    s = s.strip()
    hemisphere = s[0]

    if hemisphere in ("N", "S"):
        # Latitude: Hddmmssss (9 chars)
        deg = int(s[1:3])
        min_ = int(s[3:5])
        sec_hundredths = int(s[5:9])
        sec = sec_hundredths / 100.0
    elif hemisphere in ("E", "W"):
        # Longitude: Hdddmmssss (10 chars)
        deg = int(s[1:4])
        min_ = int(s[4:6])
        sec_hundredths = int(s[6:10])
        sec = sec_hundredths / 100.0
    else:
        return None

    value = deg + min_ / 60.0 + sec / 3600.0
    if hemisphere in ("S", "W"):
        value = -value
    return value


def parse_cifp(cifp_path: Path,
               airport_filter: str | None = None,
               ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse the CIFP file and return DataFrames for approaches, waypoints, and runways.

    If airport_filter is given (e.g. "KSBS"), only records for that airport are parsed.

    Uses arinc424 library for record parsing but accesses fields directly
    instead of going through JSON serialization (~3x faster).
    """
    if airport_filter:
        print(f"Parsing CIFP: {cifp_path} (airport={airport_filter})")
    else:
        print(f"Parsing CIFP: {cifp_path}")

    c_rows: list[dict[str, str]] = []
    g_rows: list[dict[str, str]] = []
    f_rows: list[dict[str, str]] = []

    with open(cifp_path) as f:
        for line in f:
            line = line.rstrip("\n\r")
            if len(line) < 13:
                continue
            if line[4] != "P":  # only airport records
                continue
            # Early filter by airport ICAO code (columns 7-10 in ARINC 424)
            if airport_filter and airport_filter not in line[6:11]:
                continue

            subsection = line[12]
            if subsection not in ("C", "G", "F"):
                continue

            r = arinc424.Record()
            if not r.read(line):
                continue

            # Access parsed fields directly instead of r.json() + json.loads()
            row = {field.name: field.value for field in r.fields}

            if subsection == "C":
                c_rows.append(row)
            elif subsection == "G":
                g_rows.append(row)
            else:
                f_rows.append(row)

    # --- Waypoints (subsection C) ---
    raw_wpt = pd.DataFrame(c_rows)
    wpt_df = pd.DataFrame({
        "airport": raw_wpt["Region Code"].str.strip(),
        "fix_id": raw_wpt["Waypoint Identifier"].str.strip(),
        "lat": raw_wpt["Waypoint Latitude"].map(parse_arinc_latlon),
        "lon": raw_wpt["Waypoint Longitude"].map(parse_arinc_latlon),
    }).dropna(subset=["lat", "lon"])

    # --- Runways (subsection G) ---
    raw_rwy = pd.DataFrame(g_rows)
    rwy_df = pd.DataFrame({
        "airport": raw_rwy["Airport ICAO Identifier"].str.strip(),
        "runway_id": raw_rwy["Runway Identifier"].str.strip(),
        "lat": raw_rwy["Runway Latitude"].map(parse_arinc_latlon),
        "lon": raw_rwy["Runway Longitude"].map(parse_arinc_latlon),
        "elevation_ft": pd.to_numeric(
            raw_rwy["Landing Threshold Elevation"].str.strip(), errors="coerce"),
        "tch_ft": pd.to_numeric(
            raw_rwy["Threshold Crossing Height"].str.strip(), errors="coerce"),
    }).dropna(subset=["lat", "lon"])

    # --- Approaches (subsection F) ---
    raw_apch = pd.DataFrame(f_rows)
    # Filter continuation records (keep primary records only)
    cont = raw_apch["Continuation Record No"].str.strip()
    raw_apch = raw_apch[cont.isin(("0", "1"))]

    apch_df = pd.DataFrame({
        "airport": raw_apch["Airport Identifier"].str.strip().values,
        "proc_id": raw_apch["SID/STAR/Approach Identifier"].str.strip().values,
        "route_type": raw_apch["Route Type"].str.strip().values,
        "transition": raw_apch["Transition Identifier"].str.strip().values,
        "seq_num": raw_apch["Sequence Number"].str.strip().values,
        "fix_id": raw_apch["Fix Identifier"].str.strip().values,
        "fix_section": raw_apch["Section Code (2)"].str.strip().values,
        "wpt_desc": raw_apch["Waypoint Description Code"].values,
        "path_term": raw_apch["Path and Termination"].str.strip().values,
        "alt_desc": raw_apch["Altitude Description"].str.strip().values,
        "altitude1": pd.to_numeric(
            raw_apch["Altitude"].str.strip(), errors="coerce").values,
        "altitude2": pd.to_numeric(
            raw_apch["Altitude (2)"].str.strip(), errors="coerce").values,
        "vert_angle": raw_apch["Vertical Angle"].str.strip().values,
        "qualifier1": raw_apch["Apch Route Qualifier 1"].str.strip().values,
        "qualifier2": raw_apch["Apch Route Qualifier 2"].str.strip().values,
        "center_fix": raw_apch[
            "Center Fix or TAA Procedure Turn Indicator"].str.strip().values,
    })

    print(f"  Parsed {len(wpt_df)} waypoints, {len(rwy_df)} runways, "
          f"{len(apch_df)} approach legs")
    return apch_df, wpt_df, rwy_df


# =============================================================================
# Utility Functions
# =============================================================================

def haversine_nm(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points in nautical miles.

    Works with both scalars and numpy arrays.
    """
    lat1, lon1, lat2, lon2 = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_NM * np.arcsin(np.sqrt(a))


def interpolate_point(lat1: float, lon1: float, lat2: float, lon2: float,
                      frac: float) -> tuple[float, float]:
    """Linearly interpolate between two lat/lon points (good enough for short distances)."""
    return (lat1 + frac * (lat2 - lat1), lon1 + frac * (lon2 - lon1))


# =============================================================================
# Approach Geometry Extraction (vectorized)
# =============================================================================

def _build_approach_name(proc_id: pd.Series,
                         runway_id: pd.Series) -> pd.Series:
    """Build human-readable approach names like 'RNAV (GPS) Z RWY 32'.

    Circling approaches (proc_id like 'RNV-E') become 'RNAV (GPS)-E'.
    Straight-in approaches (proc_id like 'R32-Z') become 'RNAV (GPS) Z RWY 32'.
    """
    # Use .values to get numpy arrays, avoiding PyArrow string dtype issues
    rwy_num = runway_id.str.replace("RW", "", regex=False).values.astype(str)
    suffix = (
        proc_id.str.strip()
        .str.extract(r"-?([A-EYZ]+)$", expand=False)
        .fillna("").values.astype(str)
    )
    # Circling approaches: proc_id doesn't start with R+digit (e.g. "RNV-E")
    is_straight_in = proc_id.str.match(r"^R\d").values
    straight_with_suffix = np.char.add(
        np.char.add("RNAV (GPS) ", suffix), np.char.add(" RWY ", rwy_num))
    straight_no_suffix = np.char.add("RNAV (GPS) RWY ", rwy_num)
    circling = np.char.add("RNAV (GPS)-", suffix)
    return pd.Series(np.where(
        ~is_straight_in & (suffix != ""),
        circling,
        np.where(suffix != "", straight_with_suffix, straight_no_suffix),
    ), index=proc_id.index)


def _preprocess_rnav_legs(apch_df: pd.DataFrame,
                          wpt_df: pd.DataFrame,
                          rwy_df: pd.DataFrame) -> pd.DataFrame:
    """Common preprocessing for RNAV GPS/RNP approach data.

    Filters to RNAV GPS/RNP final approach legs, assigns waypoint roles,
    keeps legs from IF through MAP (for approaches with both FAF and MAP),
    resolves runway, merges runway threshold and fix coordinate data,
    and builds approach names.

    Returns a DataFrame of legs ready for further processing by
    extract_approach_geometries() or _extract_approach_legs().
    """
    # --- Filter to RNAV GPS/RNP final approach legs ---
    # qualifier1: P = GPS, J = RNAV (RNP) — both support +V advisory guidance
    final = apch_df[
        apch_df["qualifier1"].isin(("P", "J")) & (apch_df["route_type"] == "R")
    ].copy()
    print(f"  RNAV/GPS/RNP final approach legs: {len(final)}")

    # --- Assign waypoint roles from description code ---
    # Position 3 (0-indexed): F=FAF, M=MAP, I=IF; position 2: S=Stepdown
    final["role"] = ""
    wpt3 = final["wpt_desc"].str.slice(3, 4)
    wpt2 = final["wpt_desc"].str.slice(2, 3)
    final.loc[wpt3 == "I", "role"] = "IF"
    final.loc[wpt3 == "F", "role"] = "FAF"
    final.loc[wpt3 == "M", "role"] = "MAP"
    final.loc[(wpt2 == "S") & (final["role"] == ""), "role"] = "Stepdown"

    # --- Keep only approaches with both FAF and MAP ---
    faf_seq = (
        final[final["role"] == "FAF"]
        .sort_values("seq_num")
        .drop_duplicates(["airport", "proc_id"], keep="first")
        [["airport", "proc_id", "seq_num"]]
        .rename(columns={"seq_num": "faf_seq"})
    )
    map_seq = (
        final[final["role"] == "MAP"]
        .sort_values("seq_num")
        .drop_duplicates(["airport", "proc_id"], keep="first")
        [["airport", "proc_id", "seq_num"]]
        .rename(columns={"seq_num": "map_seq"})
    )
    final = final.merge(faf_seq, on=["airport", "proc_id"], how="inner")
    final = final.merge(map_seq, on=["airport", "proc_id"], how="inner")
    final = final[final["seq_num"] <= final["map_seq"]].copy()

    # --- Resolve runway per approach ---
    # Priority: MAP fix starting with "RW", FAF center_fix starting with "RW",
    # then derive from proc_id with L/C/R suffix fallback.
    map_fixes = (
        final[final["role"] == "MAP"]
        .drop_duplicates(["airport", "proc_id"], keep="first")
        [["airport", "proc_id", "fix_id"]]
        .rename(columns={"fix_id": "map_fix"})
    )
    faf_center = (
        final[final["role"] == "FAF"]
        .sort_values("seq_num")
        .drop_duplicates(["airport", "proc_id"], keep="first")
        [["airport", "proc_id", "center_fix"]]
    )
    apch_rwy = map_fixes.merge(faf_center, on=["airport", "proc_id"], how="left")

    map_is_rwy = apch_rwy["map_fix"].str.startswith("RW")
    center_is_rwy = apch_rwy["center_fix"].str.startswith("RW").fillna(False)
    apch_rwy["runway_id"] = np.where(
        map_is_rwy, apch_rwy["map_fix"],
        np.where(center_is_rwy, apch_rwy["center_fix"], None))

    rwy_unique = rwy_df.drop_duplicates(["airport", "runway_id"], keep="first")
    missing_rwy = apch_rwy["runway_id"].isna()
    if bool(missing_rwy.any()):
        rwy_num = (
            apch_rwy.loc[missing_rwy, "proc_id"]
            .str.replace(r"^R", "", regex=True)
            .str.replace(r"[\s\-A-EYZ]+$", "", regex=True)
        )
        for suffix in ["", "L", "C", "R"]:
            if not bool(missing_rwy.any()):
                break
            candidates = pd.DataFrame({
                "airport": apch_rwy.loc[missing_rwy, "airport"].values,
                "candidate_rwy": ("RW" + rwy_num[missing_rwy] + suffix).values,
                "apch_idx": apch_rwy.index[missing_rwy],
            })
            valid = candidates.merge(
                rwy_unique[["airport", "runway_id"]],
                left_on=["airport", "candidate_rwy"],
                right_on=["airport", "runway_id"],
                how="inner",
            )
            if not valid.empty:
                apch_rwy.loc[valid["apch_idx"].values, "runway_id"] = \
                    valid["candidate_rwy"].values
                missing_rwy = apch_rwy["runway_id"].isna()

    apch_rwy = apch_rwy.dropna(subset=["runway_id"])
    final = final.merge(
        apch_rwy[["airport", "proc_id", "runway_id"]],
        on=["airport", "proc_id"], how="inner")

    # --- Merge with runway threshold data ---
    final = final.merge(
        rwy_unique[["airport", "runway_id", "lat", "lon", "elevation_ft", "tch_ft"]],
        on=["airport", "runway_id"],
        how="inner",
    ).rename(columns={
        "lat": "threshold_lat",
        "lon": "threshold_lon",
        "elevation_ft": "threshold_elev_ft",
    })
    final["tch_ft"] = final["tch_ft"].fillna(DEFAULT_TCH_FT)

    # --- Merge fix coordinates (waypoints + runways as fixes) ---
    wpt_coords = wpt_df.drop_duplicates(["airport", "fix_id"], keep="first")
    rwy_as_wpt = rwy_df[["airport", "runway_id", "lat", "lon"]].rename(
        columns={"runway_id": "fix_id"})
    all_fixes = pd.concat([
        wpt_coords[["airport", "fix_id", "lat", "lon"]],
        rwy_as_wpt,
    ]).drop_duplicates(["airport", "fix_id"], keep="first")
    final = final.merge(all_fixes, on=["airport", "fix_id"], how="left")

    # --- Compute distance from each fix to threshold ---
    has_coords = final["lat"].notna() & final["lon"].notna()
    final.loc[has_coords, "dist_to_threshold_nm"] = haversine_nm(
        final.loc[has_coords, "lat"].values,
        final.loc[has_coords, "lon"].values,
        final.loc[has_coords, "threshold_lat"].values,
        final.loc[has_coords, "threshold_lon"].values,
    )

    # --- Build approach name ---
    final["apch_name"] = _build_approach_name(final["proc_id"], final["runway_id"])

    return final


def extract_approach_geometries(apch_df: pd.DataFrame,
                                wpt_df: pd.DataFrame,
                                rwy_df: pd.DataFrame,
                                lnav_only: bool = True) -> pd.DataFrame:
    """Extract final approach geometry for each RNAV GPS approach.

    Args:
        lnav_only: if True (default), exclude approaches with published
            vertical guidance (VDA). Set False to include all approaches
            (e.g. for profile plotting).

    Returns a DataFrame with one row per approach containing:
    airport, proc_id, apch_name, faf_lat, faf_lon, faf_alt_ft,
    threshold_lat, threshold_lon, threshold_elev_ft, tch_ft, gpa_deg, distance_nm
    """
    legs = _preprocess_rnav_legs(apch_df, wpt_df, rwy_df)

    if lnav_only:
        # --- Exclude approaches with published vertical guidance ---
        # Approaches with a non-zero Vertical Angle have published VNAV
        # (LNAV/VNAV or LPV). We only want LNAV-only approaches where the
        # +V glideslope is a synthetic advisory computed by the GPS receiver.
        va_stripped = legs["vert_angle"].str.strip()
        has_vda = va_stripped.ne("") & va_stripped.ne("000") & va_stripped.ne("0")
        vda_approaches = legs.loc[has_vda, ["airport", "proc_id"]].drop_duplicates()
        legs = legs.merge(vda_approaches, on=["airport", "proc_id"],
                          how="left", indicator=True)
        legs = legs[legs["_merge"] == "left_only"].drop(columns=["_merge"])
        print(f"  After excluding published VNAV: {len(legs)} legs "
              f"({len(vda_approaches)} approaches with published VDA removed)")

    # --- Extract one row per approach from the FAF leg ---
    faf = (
        legs[legs["role"] == "FAF"]
        .sort_values("seq_num")
        .drop_duplicates(["airport", "proc_id"], keep="first")
    )
    geom = faf[["airport", "proc_id", "apch_name", "lat", "lon", "altitude1",
                "threshold_lat", "threshold_lon", "threshold_elev_ft",
                "tch_ft"]].copy()
    geom = geom.rename(columns={
        "lat": "faf_lat", "lon": "faf_lon", "altitude1": "faf_alt_ft"})

    # --- Filter and compute glidepath ---
    geom = geom.dropna(subset=[
        "faf_lat", "faf_lon", "faf_alt_ft", "threshold_elev_ft"])
    geom["distance_nm"] = haversine_nm(
        geom["faf_lat"].values, geom["faf_lon"].values,
        geom["threshold_lat"].values, geom["threshold_lon"].values,
    )
    geom = geom[geom["distance_nm"] * FEET_PER_NM > 100].copy()

    geom["alt_drop"] = (geom["faf_alt_ft"] - geom["threshold_elev_ft"]
                        - geom["tch_ft"])
    geom = geom[geom["alt_drop"] > 0].copy()
    geom["gpa_deg"] = np.degrees(
        np.arctan2(geom["alt_drop"].values,
                   geom["distance_nm"].values * FEET_PER_NM)
    )

    geom = geom[
        ["airport", "proc_id", "apch_name", "faf_lat", "faf_lon", "faf_alt_ft",
         "threshold_lat", "threshold_lon", "threshold_elev_ft", "tch_ft",
         "gpa_deg", "distance_nm"]
    ].reset_index(drop=True)

    print(f"  Extracted geometry for {len(geom)} approaches")
    return geom


# =============================================================================
# High-level API
# =============================================================================

PLATE_DB_URL = (
    "https://github.com/ammaraskar/faa-instrument-approach-db"
    "/releases/download/251225/approaches.json"
)
AIRPORTS_CSV_URL = (
    "https://davidmegginson.github.io/ourairports-data/airports.csv"
)


class ApproachDatabase:
    """Unified access to RNAV approach data from CIFP and OCR'd plate minimums.

    Parses the CIFP lazily on first access and caches the result so that
    get_approaches() and get_approach_legs() share a single parse.
    """

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._plate_db: dict | None = None
        self._airport_db: dict[str, dict] | None = None
        self._cifp_airport: str | None = object()  # sentinel: not yet parsed
        self._cifp_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None

    def _ensure_cifp(self, airport: str | None = None
                     ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Download CIFP if needed and parse it, caching the result.

        If the cached parse used the same airport filter, returns the cache.
        Otherwise re-parses (this only happens if callers disagree on the filter,
        which shouldn't occur in normal usage).
        """
        if self._cifp_data is not None and self._cifp_airport == airport:
            return self._cifp_data
        cifp_path = download_cifp(self.data_dir)
        self._cifp_data = parse_cifp(cifp_path, airport_filter=airport)
        self._cifp_airport = airport
        return self._cifp_data

    def get_approaches(self, airport: str | None = None,
                       lnav_only: bool = True) -> pd.DataFrame:
        """Load approach geometries.

        Downloads CIFP if not already cached. Returns a DataFrame with one row
        per approach containing FAF/threshold geometry, glidepath angle, and
        approach name.

        Args:
            airport: ICAO code to filter (e.g. "KSBS"); None for all US.
            lnav_only: if True (default), only LNAV-only approaches (no
                published VDA). Set False to include all RNAV GPS approaches.
        """
        apch_df, wpt_df, rwy_df = self._ensure_cifp(airport)
        return extract_approach_geometries(apch_df, wpt_df, rwy_df,
                                           lnav_only=lnav_only)

    def get_approach_legs(self, airport: str | None = None) -> pd.DataFrame:
        """Load detailed approach legs with coordinates for profile plotting.

        Returns all RNAV (GPS/RNP) approach legs from IF through MAP,
        with fix coordinates, altitude constraints, and runway data.
        Includes both LNAV-only and published VNAV approaches.
        """
        apch_df, wpt_df, rwy_df = self._ensure_cifp(airport)
        return _extract_approach_legs(apch_df, wpt_df, rwy_df)

    def get_terrain(self, srtm1: bool = False) -> "Terrain":
        """Create a terrain elevation engine.

        Args:
            srtm1: if True, use srtm.py library for 30m SRTM1 resolution;
                   otherwise use local SRTM3 90m tiles (faster for bulk queries)
        """
        if srtm1:
            return SRTMLibTerrain(srtm1=True)
        return SRTMTerrain(self.data_dir / "srtm3")

    def get_minimums(self, airport: str,
                     approach_name: str) -> dict[str, int] | None:
        """Look up OCR'd minimums for an approach.

        Args:
            airport: ICAO code (e.g. "KSBS")
            approach_name: as built by _build_approach_name
                           (e.g. "RNAV (GPS) Z RWY 32")

        Returns dict mapping minimums_type to Cat A altitude in feet,
        e.g. {"LNAV": 9100, "CIRCLING": 9100}, or None if not found.
        """
        db = self._ensure_plate_db()
        airports = db.get("airports", {})
        apt_data = airports.get(airport)
        if apt_data is None:
            return None

        for apch in apt_data.get("approaches", []):
            if apch.get("name") == approach_name:
                result: dict[str, int] = {}
                for m in apch.get("minimums", []):
                    mtype = m.get("minimums_type", "")
                    cat_a = m.get("cat_a")
                    if isinstance(cat_a, dict):
                        alt_str = cat_a.get("altitude")
                        if alt_str is not None:
                            try:
                                result[mtype] = int(alt_str)
                            except (ValueError, TypeError):
                                pass
                    elif isinstance(cat_a, str) and cat_a == "NA":
                        continue
                return result if result else None
        return None

    def get_airport_info(self, icao: str) -> dict[str, str]:
        """Look up airport name and state from OurAirports database.

        Returns dict with keys: name, state, municipality.
        Values are empty strings if not found.
        """
        db = self._ensure_airport_db()
        empty = {"name": "", "state": "", "municipality": ""}
        result = db.get(icao)
        if result is None:
            # OurAirports uses K-prefixed idents for small US airports
            result = db.get("K" + icao)
        return result if result is not None else empty

    def _ensure_plate_db(self) -> dict:
        """Download (if needed) and load the OCR'd approach plate JSON."""
        if self._plate_db is not None:
            return self._plate_db

        plate_path = self.data_dir / "faa_approach_plates.json"
        if not plate_path.exists():
            print("Downloading approach plate database...")
            r = requests.get(PLATE_DB_URL, timeout=120)
            r.raise_for_status()
            plate_path.write_bytes(r.content)
            print(f"  Saved {len(r.content) / 1024 / 1024:.1f} MB to {plate_path}")

        with open(plate_path) as f:
            self._plate_db = json.load(f)
        return self._plate_db

    def _ensure_airport_db(self) -> dict[str, dict]:
        """Download (if needed) and load the OurAirports CSV as a lookup dict."""
        if self._airport_db is not None:
            return self._airport_db

        csv_path = self.data_dir / "airports.csv"
        if not csv_path.exists():
            print("Downloading OurAirports database...")
            r = requests.get(AIRPORTS_CSV_URL, timeout=60)
            r.raise_for_status()
            csv_path.write_bytes(r.content)
            print(f"  Saved {len(r.content) / 1024 / 1024:.1f} MB to {csv_path}")

        self._airport_db = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                region = row.get("iso_region", "")
                state = region.split("-")[1] if "-" in region else ""
                info = {
                    "name": row.get("name", ""),
                    "state": state,
                    "municipality": row.get("municipality", ""),
                }
                # Index by ident, icao_code, and gps_code so lookups
                # work regardless of which identifier CIFP uses
                for key_col in ("ident", "icao_code", "gps_code"):
                    key = row.get(key_col, "").strip()
                    if key and key not in self._airport_db:
                        self._airport_db[key] = info
        return self._airport_db


def _extract_approach_legs(apch_df: pd.DataFrame,
                           wpt_df: pd.DataFrame,
                           rwy_df: pd.DataFrame) -> pd.DataFrame:
    """Extract detailed approach legs from IF through MAP for profile plotting.

    Returns all RNAV (GPS/RNP) approach legs with fix coordinates,
    altitude constraints, and runway data. Includes both LNAV-only
    and published VNAV approaches.
    """
    final = _preprocess_rnav_legs(apch_df, wpt_df, rwy_df)

    result = final[
        ["airport", "proc_id", "apch_name", "fix_id", "seq_num", "role",
         "lat", "lon", "altitude1", "alt_desc", "vert_angle",
         "dist_to_threshold_nm", "threshold_lat", "threshold_lon",
         "threshold_elev_ft", "tch_ft", "runway_id"]
    ].sort_values(["airport", "proc_id", "seq_num"]).reset_index(drop=True)

    n_approaches = result.groupby(["airport", "proc_id"]).ngroups
    print(f"  Loaded {len(result)} legs for {n_approaches} approaches")
    return result
