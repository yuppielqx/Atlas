"""
Fetch historical weather data for METR-LA sensor regions from Open-Meteo Archive API.

Clusters sensors into K regions via K-means on lat/lon, then queries Open-Meteo
at each cluster centroid for 2012-03-01 to 2012-06-30.

Output: weather.csv with columns:
    datetime, temperature_2m, precipitation, weather_code, region_id, latitude, longitude

Also saves: sensor_weather_map.csv mapping sensor_id -> region_id.

Usage:
    python scripts/fetch_weather.py \
        --dataset metr-la \
        --data_dir ./datasets \
        --n_regions 5 \
        --output_dir ./output
"""

import os
import sys
import argparse
import csv
import time
import datetime as dt
import numpy as np
import urllib.request
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch historical weather from Open-Meteo API")
    parser.add_argument("--dataset", type=str, default="metr-la",
                        choices=["metr-la", "pems-bay"])
    parser.add_argument("--data_dir", type=str, default="./datasets")
    parser.add_argument("--n_regions", type=int, default=5,
                        help="Number of K-means clusters for weather regions")
    parser.add_argument("--start_date", type=str, default="2012-03-01")
    parser.add_argument("--end_date", type=str, default="2012-06-30")
    parser.add_argument("--output_dir", type=str, default="./output")
    return parser.parse_args()


def load_sensor_locations(dataset_name, data_dir):
    """Load sensor_id -> (lat, lon) mapping from graph_sensor_locations.csv."""
    dataset_dirs = {"metr-la": "METR-LA", "pems-bay": "PEMS-BAY"}
    loc_csv = os.path.join(data_dir, dataset_dirs[dataset_name],
                           "graph_sensor_locations.csv")

    sensors = {}  # index -> {sensor_id, lat, lon}
    with open(loc_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["index"])
            sensors[idx] = {
                "sensor_id": int(row["sensor_id"]),
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
            }
    return sensors


def cluster_sensors(sensors, n_regions, random_state=42):
    """K-means clustering of sensors by (lat, lon)."""
    from sklearn.cluster import KMeans

    ids = sorted(sensors.keys())
    coords = np.array([[sensors[i]["lat"], sensors[i]["lon"]] for i in ids])

    # Use more clusters if there are few regions relative to spread
    k = min(n_regions, len(ids))
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(coords)
    centroids = km.cluster_centers_

    # Build mapping: sensor_index -> region_id
    sensor_region = {}
    for i, idx in enumerate(ids):
        sensor_region[sensors[idx]["sensor_id"]] = int(labels[i])

    regions = {}
    for r in range(k):
        regions[r] = {"lat": float(centroids[r, 0]), "lon": float(centroids[r, 1]),
                      "n_sensors": int((labels == r).sum())}

    return sensor_region, regions


def build_openmeteo_url(lat, lon, start_date, end_date):
    """Build Open-Meteo Archive API URL for historical weather."""
    base = "https://archive-api.open-meteo.com/v1/archive"
    params = (
        f"latitude={lat:.4f}&longitude={lon:.4f}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,precipitation,weather_code"
        f"&timezone=America/Los_Angeles"
    )
    return f"{base}?{params}"


def fetch_weather_for_region(region_id, lat, lon, start_date, end_date):
    """Query Open-Meteo API and parse hourly data for one region."""
    url = build_openmeteo_url(lat, lon, start_date, end_date)
    print(f"  Region {region_id} ({lat:.4f}, {lon:.4f}): querying...")

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"    ERROR: {e}")
        return None

    if "hourly" not in data:
        print(f"    ERROR: no 'hourly' in response: {data}")
        return None

    hourly = data["hourly"]
    times = hourly["time"]
    temps = hourly.get("temperature_2m", [])
    precip = hourly.get("precipitation", [])
    codes = hourly.get("weather_code", [])

    rows = []
    for i, t_str in enumerate(times):
        rows.append({
            "datetime": t_str,
            "temperature_2m": temps[i] if i < len(temps) else None,
            "precipitation": precip[i] if i < len(precip) else None,
            "weather_code": codes[i] if i < len(codes) else None,
            "region_id": region_id,
            "latitude": lat,
            "longitude": lon,
        })

    print(f"    Got {len(rows)} hourly records "
          f"({times[0]} to {times[-1]})")
    return rows


def main():
    args = parse_args()

    # 1. Load sensor locations
    sensors = load_sensor_locations(args.dataset, args.data_dir)
    print(f"Loaded {len(sensors)} sensors from {args.dataset}")

    # 2. Cluster sensors
    sensor_region, regions = cluster_sensors(sensors, args.n_regions)
    print(f"Clustered into {len(regions)} regions:")
    for r, info in regions.items():
        print(f"  Region {r}: ({info['lat']:.4f}, {info['lon']:.4f}) "
              f"- {info['n_sensors']} sensors")

    # 3. Fetch weather for each region (rate-limited)
    all_rows = []
    for r, info in regions.items():
        rows = fetch_weather_for_region(
            r, info["lat"], info["lon"],
            args.start_date, args.end_date)
        if rows:
            all_rows.extend(rows)
        if r < len(regions) - 1:
            time.sleep(1.0)  # Be polite to the free API

    if not all_rows:
        print("ERROR: No weather data fetched!")
        sys.exit(1)

    # 4. Save weather CSV
    os.makedirs(args.output_dir, exist_ok=True)
    weather_path = os.path.join(args.output_dir, "weather.csv")
    fieldnames = ["datetime", "temperature_2m", "precipitation",
                  "weather_code", "region_id", "latitude", "longitude"]
    with open(weather_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved weather data ({len(all_rows)} records) to: {weather_path}")

    # 5. Save sensor -> region mapping
    map_path = os.path.join(args.output_dir, "sensor_weather_map.csv")
    with open(map_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sensor_id", "region_id"])
        for sid, rid in sorted(sensor_region.items()):
            writer.writerow([sid, rid])
    print(f"Saved sensor-region mapping to: {map_path}")

    # 6. Print summary
    print(f"\nCoverage: {args.start_date} to {args.end_date}")
    print(f"Regions: {len(regions)}")
    unique_dates = set(r["datetime"][:10] for r in all_rows)
    print(f"Unique dates: {len(unique_dates)}")


if __name__ == "__main__":
    main()
