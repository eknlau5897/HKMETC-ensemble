#!/bin/bash

# Working directory setup
WORKSPACE="/Users/eknlau/VS_code/HKMETC-ensemble/"
if [ -d "$WORKSPACE" ]; then
    cd "$WORKSPACE" || exit 1
fi

echo "========================================================="
echo " Starting Unified HKMETC Engine Pipeline"
echo " (ECMWF, AIFS, GENC, FNV3.0, FNV3.1, WNC/FNV3.2)"
echo "========================================================="

while true; do
    echo "--- Cycle Started at $(date) ---"

    # =========================================================
    # 1. ECMWF & AIFS PIPELINE
    # =========================================================
    for MODEL in "ECMWF" "AIFS"; do
        echo "--> Processing Model: $MODEL"
        python3.11 - "$MODEL" << 'EOF'
import os
import sys
from datetime import datetime, timedelta, timezone
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from ecmwf.opendata import Client
import earthkit.data

model_type = sys.argv[1]

# Compute closest operational runtime (UTC)
now_utc = datetime.now(timezone.utc)
if now_utc.hour >= 20:
    init_date_dt = now_utc
    init_time = 12
elif now_utc.hour >= 14:
    init_date_dt = now_utc
    init_time = 6
elif now_utc.hour >= 8:
    init_date_dt = now_utc
    init_time = 0
elif now_utc.hour >= 2:
    init_date_dt = now_utc - timedelta(days=1)
    init_time = 18
else:
    init_date_dt = now_utc - timedelta(days=1)
    init_time = 12

# Map extent set to [100, 140, 7, 36]
extent = [100, 140, 7, 36]

# Set forecast hours
if model_type == "AIFS":
    forecast_hours = 360
    client_kwargs = {"source": "ecmwf", "model": "aifs-ens"}
    line_color, line_style = '#1e88e5', '--'
    title_prefix, title_color = "AIFS", '#0d47a1'
    subtitle = f"{forecast_hours}-hour Forecast (aifs-ens)"
else:
    forecast_hours = 144 if init_time in [6, 18] else 360
    client_kwargs = {"source": "ecmwf", "model": "ifs"}
    line_color, line_style = '#546e7a', '-'
    title_prefix, title_color = "ECMWF", '#1a237e'
    subtitle = f"{forecast_hours}-hour Forecast"

init_date = init_date_dt.strftime("%Y-%m-%d")
date_folder = init_date_dt.strftime("%Y%m%d")
time_str = f"{init_time:02d}Z"

path = f"/Users/eknlau/VS_code/HKMETC-ensemble/ensemble-track/wp/{model_type}/{date_folder}/{time_str}/"
os.makedirs(path, exist_ok=True)

target_bufr = os.path.join(path, f"{model_type.lower()}-{init_date}-{time_str}.bufr")
csv_nwp = os.path.join(path, f"{model_type.lower()}-{init_date}-{time_str}-NWP.csv")
output_png = os.path.join(path, "240.png")

print(f"[{datetime.now()}] Fetching {model_type} ({time_str}) for {forecast_hours}h...")

try:
    client = Client(**client_kwargs)
    client.retrieve(date=init_date, time=init_time, type="tf", stream="enfo", step=forecast_hours, target=target_bufr)
except Exception as e:
    print(f"Data not ready or download failed for {model_type}: {e}")
    sys.exit(0)

try:
    ds = earthkit.data.from_source("file", target_bufr)
    df = ds.to_pandas(
        columns=["stormIdentifier", "ensembleMemberNumber", "typicalDate", "typicalTime", 
                 "year", "month", "day", "hour", "latitude", "longitude", "pressureReducedToMeanSeaLevel"],
        filters={"meteorologicalAttributeSignificance": 1}, required_columns=True
    )
    df = df.dropna(subset=['year', 'month', 'day', 'hour', 'longitude', 'latitude']).copy()

    # Robust zero-padding for typicalTime formatting
    b_date = df['typicalDate'].astype(int).astype(str)
    b_time = df['typicalTime'].astype(int).astype(str).str.zfill(6)
    df['base_dt'] = pd.to_datetime(b_date + b_time, format='%Y%m%d%H%M%S', errors='coerce')
    
    df['valid_dt'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df['pressure'] = df['pressureReducedToMeanSeaLevel'] / 100.0
    df = df.rename(columns={"stormIdentifier": "track", "ensembleMemberNumber": "sample", "longitude": "lon", "latitude": "lat"})

    df.loc[df['lon'] < 0, 'lon'] += 360

    df_nwp = df[df['track'].astype(str).str.endswith('W', na=False)].copy()
    df_nwp.to_csv(csv_nwp, index=False, encoding='utf-8-sig')

    if df_nwp.empty:
        print(f"No Western Pacific tracks found for {model_type}.")
        sys.exit(0)

    base_time_str = df_nwp['base_dt'].iloc[0].strftime('%Y-%m-%d %H:%MZ')

    bounds = [900, 915, 930, 945, 960, 970, 980, 990, 1000, 1010]
    colors = ['#4a148c', '#880e4f', '#b71c1c', '#e65100', '#ff8f00', '#fbc02d', '#4db6ac', '#0277bd', '#808080']
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=(12, 9), dpi=120, facecolor='#ffffff')
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor='#f9fbfc', zorder=0)
    ax.add_feature(cfeature.LAND, facecolor='#f0f2f0', edgecolor='#bcbcbc', linewidth=0.5, zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor='#454545', zorder=2)
    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.4, edgecolor='#95a5a6', zorder=2)

    gl = ax.gridlines(draw_labels=True, linestyle=':', alpha=0.4, color='#7f8c8d', zorder=3)
    gl.top_labels, gl.right_labels = False, False

    for _, group in df_nwp.groupby(['track', 'sample']):
        ax.plot(group['lon'], group['lat'], color=line_color, linestyle=line_style, linewidth=0.5, alpha=0.3, transform=ccrs.PlateCarree(), zorder=4)

    sc = ax.scatter(df_nwp['lon'], df_nwp['lat'], edgecolors=cmap(norm(df_nwp['pressure'])), facecolors='none', s=20, linewidths=1.2, alpha=0.9, transform=ccrs.PlateCarree(), zorder=5)

    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, pad=0.03, fraction=0.04, aspect=30)
    cbar.set_label('Minimum Sea Level Pressure (hPa)', fontsize=10, labelpad=10, fontweight='bold')

    plt.title(f"{title_prefix} Ensemble Tracks - HKMETC", fontsize=16, fontweight='bold', pad=40, color=title_color)
    plt.text(0.5, 1.05, subtitle, transform=ax.transAxes, ha='center', fontsize=13, fontweight='bold', color='#333333')
    plt.text(0.5, 1.02, f"Initial Time: {base_time_str}", transform=ax.transAxes, ha='center', fontsize=11, color='#546e7a')

    plt.savefig(output_png, bbox_inches='tight')
    plt.close(fig)
    print(f"Successfully generated {model_type} plot asset.")

except Exception as e:
    print(f"Error processing {model_type}: {e}")
EOF
    done

    # =========================================================
    # 2. GOOGLE WEATHER LAB PIPELINE (GENC, FNV3.0, FNV3.1 & WNC)
    # =========================================================
    echo "--> Running Google Weather Lab Engine (GENC, FNV3.0, FNV3.1 & WNC/FNV3.2)..."
    python3.11 << 'EOF'
import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

base_path = "/Users/eknlau/VS_code/HKMETC-ensemble/ensemble-track/wp"

models = {
    "FNV3":   {"url_keys": ["FNV3P0"],            "title": "FNV3.0 Ensemble Tracks - HKMETC"},
    "FNV3.1": {"url_keys": ["FNV3P1", "FNV3P1"], "title": "FNV3.1 Ensemble Tracks - HKMETC"},
    "WNC":    {"url_keys": ["FNV3P2", "FNV3P2"],   "title": "FNV3.2 (WNC) Ensemble Tracks - HKMETC"}
}

bounds = [900, 915, 930, 945, 960, 970, 980, 990, 1000, 1010]
cmap = mcolors.ListedColormap(['#4a148c', '#880e4f', '#b71c1c', '#e65100', '#ff8f00', '#fbc02d', '#4db6ac', '#0277bd', '#808080'])
norm = mcolors.BoundaryNorm(bounds, cmap.N)

for model_name, cfg in models.items():
    print(f"[{datetime.now()}] Initializing search engine for {model_name}...")
    success = False
    
    for lookback_hours in range(0, 121, 6):
        now_utc = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        
        if now_utc.hour >= 19:
            init_date_dt = now_utc
            init_time = 12
        elif now_utc.hour >= 13:
            init_date_dt = now_utc
            init_time = 6
        elif now_utc.hour >= 7:
            init_date_dt = now_utc
            init_time = 0
        elif now_utc.hour >= 1:
            init_date_dt = now_utc - timedelta(days=1)
            init_time = 18
        else:
            init_date_dt = now_utc - timedelta(days=1)
            init_time = 12

        yyyy = init_date_dt.strftime("%Y")
        mm = init_date_dt.strftime("%m")
        dd = init_date_dt.strftime("%d")
        date_folder = f"{yyyy}{mm}{dd}"
        cycle_str = f"{init_time:02d}Z"
        
        time_stamps = [
            f"{yyyy}_{mm}_{dd}T{init_time:02d}_00", # T06_00
            f"{yyyy}_{mm}_{dd}T{init_time}_00"      # T6_00
        ]
        
        r = None
        target_url = ""
        
        for url_id in cfg["url_keys"]:
            for ts in time_stamps:
                candidate_url = f"https://deepmind.google.com/science/weatherlab/download/cyclones/{url_id}/ensemble/cyclogenesis/csv/{url_id}_{ts}_cyclogenesis.csv"
                try:
                    res = requests.get(candidate_url, timeout=8)
                    if res.status_code == 200 and not res.text.lstrip().startswith("<!DOCTYPE html>") and "<html" not in res.text.lower():
                        r = res
                        target_url = candidate_url
                        break
                except Exception:
                    continue
            if r is not None:
                break

        if r is None:
            print(f"    [DEBUG] Skipping {model_name} ({date_folder} {cycle_str}): HTTP 404")
            continue
            
        try:
            path_2 = f"{base_path}/{model_name}"
            run_dir = os.path.join(path_2, date_folder, cycle_str)
            os.makedirs(run_dir, exist_ok=True)
            
            local_csv_path = os.path.join(run_dir, f"{model_name.lower().replace('.', '')}-unpaired-NWP.csv")
            archive_png = os.path.join(run_dir, "240.png")            
            latest_png = os.path.join(path_2, "240.png")
            
            with open(local_csv_path, 'wb') as f:
                f.write(r.content)
                
            df_2 = pd.read_csv(local_csv_path, comment='#')
            if df_2.empty:
                print(f"    [DEBUG] CSV empty for {model_name} ({date_folder} {cycle_str})")
                continue
                
            if 'minimum_sea_level_pressure_hpa' in df_2.columns:
                df_2 = df_2.rename(columns={'minimum_sea_level_pressure_hpa': 'pressure'})
            
            if 'lon' in df_2.columns:
                df_2.loc[df_2['lon'] < 0, 'lon'] += 360
            
            df_2 = df_2.sort_values(by=['track_id', 'sample', 'valid_time'])
            base_time_str = f"{yyyy}-{mm}-{dd} {cycle_str}"

            fig = plt.figure(figsize=(12, 9), dpi=100)
            ax_2 = plt.axes(projection=ccrs.PlateCarree())
            # Map extent set to [100, 140, 7, 36]
            ax_2.set_extent([100, 140, 7, 36], crs=ccrs.PlateCarree())

            ax_2.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=2)
            ax_2.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.4, zorder=2)
            ax_2.add_feature(cfeature.LAND, facecolor='#f5f5f5', zorder=1)

            gl = ax_2.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False

            for _, group in df_2.groupby(['track_id', 'sample']):
                ax_2.plot(group['lon'], group['lat'], color='gray', linewidth=0.6, alpha=0.2, transform=ccrs.PlateCarree(), zorder=3)

            sc = ax_2.scatter(
                df_2['lon'], df_2['lat'], 
                edgecolors=cmap(norm(df_2['pressure'])), 
                facecolors='none', s=15, linewidths=1.2, alpha=0.9, 
                transform=ccrs.PlateCarree(), zorder=4
            )

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            cbar = plt.colorbar(sm, ax=ax_2, pad=0.04, fraction=0.05, aspect=25)
            cbar.set_label('Minimum Sea Level Pressure (hPa)', rotation=270, labelpad=20)
            cbar.set_ticks(bounds) 

            plt.title(cfg["title"], fontsize=16, fontweight='bold', pad=40, color='#1a237e')
            plt.text(0.5, 1.05, "360-hour Forecast", transform=ax_2.transAxes, ha='center', fontsize=13, fontweight='bold', color='#333333')
            plt.text(0.5, 1.02, f"Initial Time: {base_time_str}", transform=ax_2.transAxes, ha='center', fontsize=11, color='#546e7a')
            
            plt.savefig(archive_png, bbox_inches='tight') 
            plt.savefig(latest_png, bbox_inches='tight')  
            plt.close(fig)
            
            print(f"--> [SUCCESS] Processed {model_name} ({cycle_str}) from {target_url}!")
            success = True
            break 
            
        except Exception as e:
            print(f"    Internal cycle error for {model_name} {cycle_str}: {e}")
            continue
            
    if not success:
        print(f"[WARN] Failed to find any valid recent cycles for {model_name}.")
EOF

    # =========================================================
    # 3. GIT AUTOMATED SYNC
    # =========================================================
    echo "Pushing updates to GitHub main branch..."
    git add .
    if ! git diff-index --quiet HEAD --; then
        git commit -m "Automated Sync: HKMETC Tracks Updated ($(date -u +'%Y-%m-%d %H:%MZ'))"
        git push origin main
    else
        echo "No changes detected in working tree. Skipping git commit/push."
    fi

    # =========================================================
    # 4. DYNAMIC SLEEP CALCULATION (Snaps to 02, 08, 14, 20 UTC)
    # =========================================================
    CURRENT_HOUR=$(date -u +%-H)
    CURRENT_MIN=$(date -u +%-M)
    CURRENT_SEC=$(date -u +%-S)

    if [ "$CURRENT_HOUR" -lt 2 ]; then NEXT_TARGET=2
    elif [ "$CURRENT_HOUR" -lt 8 ]; then NEXT_TARGET=8
    elif [ "$CURRENT_HOUR" -lt 14 ]; then NEXT_TARGET=14
    elif [ "$CURRENT_HOUR" -lt 20 ]; then NEXT_TARGET=20
    else NEXT_TARGET=26 # 02:00Z next day
    fi

    HOURS_TO_WAIT=$((NEXT_TARGET - CURRENT_HOUR - 1))
    MINS_TO_WAIT=$((59 - CURRENT_MIN))
    SECS_TO_WAIT=$((60 - CURRENT_SEC))

    TOTAL_SLEEP=$(( (HOURS_TO_WAIT * 3600) + (MINS_TO_WAIT * 60) + SECS_TO_WAIT ))

    echo "========================================================="
    echo "Cycle completed at $(date -u)."
    echo "Next run target: $((NEXT_TARGET % 24)):00Z."
    echo "Sleeping for $TOTAL_SLEEP seconds..."
    echo "========================================================="
    
    sleep $TOTAL_SLEEP
done