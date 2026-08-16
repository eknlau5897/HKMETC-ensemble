#!/bin/bash

echo "========================================================="
echo " Starting Unified GHMWS Engine Pipeline (ECMWF & AIFS)"
echo "========================================================="

while true; do
    echo "--- Cycle Started at $(date) ---"

    # =========================================================
    # 1. ECMWF PIPELINE (Dynamic: 6Z/18Z -> 144h, 0Z/12Z -> 360h)
    # =========================================================
    python3.11 - "ECMWF" << 'EOF'
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from ecmwf.opendata import Client
import earthkit.data

model_type = sys.argv[1]

# Compute closest operational runtime
now_utc = datetime.utcnow()
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

# ECMWF specific conditional forecast length
if init_time in [6, 18]:
    forecast_hours = 144
else:
    forecast_hours = 360

init_date = init_date_dt.strftime("%Y-%m-%d")
date_folder = init_date_dt.strftime("%Y%m%d")
time_str = f"{init_time:02d}Z"

path = f"/Users/eknlau/VS_code/GHMWS/ensemble-track/wp/{model_type}/{date_folder}/{time_str}/"
if not os.path.exists(path):
    os.makedirs(path)

target_bufr = os.path.join(path, f"ifs-{init_date}-{time_str}.bufr")
csv_nwp = os.path.join(path, f"ifs-{init_date}-{time_str}-NWP.csv")
client_kwargs = {"source": "ecmwf", "model": "ifs"}
line_color, line_style = '#546e7a', '-'
title_prefix, title_color = "ECMWF", '#1a237e'
subtitle = f"{forecast_hours}-hour Forecast"

# Reverted strictly back to 240.png as requested
output_png = os.path.join(path, "240.png")
print(f"[{datetime.now()}] Processing {model_type} ({time_str}) for {forecast_hours} hours -> Saving to 240.png...")

try:
    client = Client(**client_kwargs)
    client.retrieve(date=init_date, time=init_time, type="tf", stream="enfo", step=forecast_hours, target=target_bufr)
except Exception as e:
    print(f"Data not ready or download failed for {model_type}: {e}")
    sys.exit(0)

ds = earthkit.data.from_source("file", target_bufr)
df = ds.to_pandas(
    columns=["stormIdentifier", "ensembleMemberNumber", "typicalDate", "typicalTime", 
             "year", "month", "day", "hour", "latitude", "longitude", "pressureReducedToMeanSeaLevel"],
    filters={"meteorologicalAttributeSignificance": 1}, required_columns=True
)
df = df.dropna(subset=['year', 'month', 'day', 'hour', 'longitude', 'latitude']).copy()

b_date = df['typicalDate'].astype(int).astype(str)
b_time = df['typicalTime'].astype(int).astype(str).str.zfill(4) 
df['base_dt'] = pd.to_datetime((b_date + b_time).str[:12], format='%Y%m%d%H%M')
df['valid_dt'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
df['pressure'] = df['pressureReducedToMeanSeaLevel'] / 100.0
df = df.rename(columns={"stormIdentifier": "track", "ensembleMemberNumber": "sample", "longitude": "lon", "latitude": "lat"})

# --- ADJUST NEGATIVE LONGITUDE HERE ---
df.loc[df['lon'] < 0, 'lon'] += 360

df_nwp = df[df['track'].astype(str).str.endswith('W', na=False)].copy()
df_nwp.to_csv(csv_nwp, index=False, encoding='utf-8-sig')

if df_nwp.empty:
    print(f"No tracks found for {model_type}.")
    sys.exit(0)

base_time_str = df_nwp['base_dt'].iloc[0].strftime('%Y-%m-%d %H:%MZ')

bounds = [900, 915, 930, 945, 960, 970, 980, 990, 1000, 1010]
colors = ['#4a148c', '#880e4f', '#b71c1c', '#e65100', '#ff8f00', '#fbc02d', '#4db6ac', '#0277bd', '#808080']
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

fig = plt.figure(figsize=(12, 9), dpi=120, facecolor='#ffffff')
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([100,160,7,45], crs=ccrs.PlateCarree())
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

plt.title(f"{title_prefix} Ensemble Tracks - GHMWS", fontsize=16, fontweight='bold', pad=40, color=title_color)
plt.text(0.5, 1.05, subtitle, transform=ax.transAxes, ha='center', fontsize=13, fontweight='bold', color='#333333')
plt.text(0.5, 1.02, f"Initial Time: {base_time_str}", transform=ax.transAxes, ha='center', fontsize=11, color='#546e7a')

plt.savefig(output_png, bbox_inches='tight')
plt.close(fig)
print(f"Successfully generated {model_type} plot asset.")
EOF

    echo "---------------------------------------------------------"

    # =========================================================
    # 2. AIFS PIPELINE (Static: Always 360h)
    # =========================================================
    python3.11 - "AIFS" << 'EOF'
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from ecmwf.opendata import Client
import earthkit.data

model_type = sys.argv[1]

# Compute closest operational runtime
now_utc = datetime.utcnow()
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

# AIFS always uses 360 hours regardless of init_time
forecast_hours = 360

init_date = init_date_dt.strftime("%Y-%m-%d")
date_folder = init_date_dt.strftime("%Y%m%d")
time_str = f"{init_time:02d}Z"

path = f"/Users/eknlau/VS_code/GHMWS/ensemble-track/wp/{model_type}/{date_folder}/{time_str}/"
if not os.path.exists(path):
    os.makedirs(path)

target_bufr = os.path.join(path, f"aifs-{init_date}-{time_str}.bufr")
csv_nwp = os.path.join(path, f"aifs-{init_date}-{time_str}-NWP.csv")
client_kwargs = {"source": "ecmwf", "model": "aifs-ens"}
line_color, line_style = '#1e88e5', '--'
title_prefix, title_color = "AIFS", '#0d47a1'
subtitle = f"{forecast_hours}-hour Forecast (aifs-ens)"

# Reverted strictly back to 240.png as requested
output_png = os.path.join(path, "240.png")
print(f"[{datetime.now()}] Processing {model_type} ({time_str}) for {forecast_hours} hours -> Saving to 240.png...")

try:
    client = Client(**client_kwargs)
    client.retrieve(date=init_date, time=init_time, type="tf", stream="enfo", step=forecast_hours, target=target_bufr)
except Exception as e:
    print(f"Data not ready or download failed for {model_type}: {e}")
    sys.exit(0)

ds = earthkit.data.from_source("file", target_bufr)
df = ds.to_pandas(
    columns=["stormIdentifier", "ensembleMemberNumber", "typicalDate", "typicalTime", 
             "year", "month", "day", "hour", "latitude", "longitude", "pressureReducedToMeanSeaLevel"],
    filters={"meteorologicalAttributeSignificance": 1}, required_columns=True
)
df = df.dropna(subset=['year', 'month', 'day', 'hour', 'longitude', 'latitude']).copy()

b_date = df['typicalDate'].astype(int).astype(str)
b_time = df['typicalTime'].astype(int).astype(str).str.zfill(4) 
df['base_dt'] = pd.to_datetime((b_date + b_time).str[:12], format='%Y%m%d%H%M')
df['valid_dt'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
df['pressure'] = df['pressureReducedToMeanSeaLevel'] / 100.0
df = df.rename(columns={"stormIdentifier": "track", "ensembleMemberNumber": "sample", "longitude": "lon", "latitude": "lat"})

# --- ADJUST NEGATIVE LONGITUDE HERE ---
df.loc[df['lon'] < 0, 'lon'] += 360

df_nwp = df[df['track'].astype(str).str.endswith('W', na=False)].copy()
df_nwp.to_csv(csv_nwp, index=False, encoding='utf-8-sig')

if df_nwp.empty:
    print(f"No tracks found for {model_type}.")
    sys.exit(0)

base_time_str = df_nwp['base_dt'].iloc[0].strftime('%Y-%m-%d %H:%MZ')

bounds = [900, 915, 930, 945, 960, 970, 980, 990, 1000, 1010]
colors = ['#4a148c', '#880e4f', '#b71c1c', '#e65100', '#ff8f00', '#fbc02d', '#4db6ac', '#0277bd', '#808080']
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

fig = plt.figure(figsize=(12, 9), dpi=120, facecolor='#ffffff')
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([100, 180, 0, 60], crs=ccrs.PlateCarree())
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

plt.title(f"{title_prefix} Ensemble Tracks - GHMWS", fontsize=16, fontweight='bold', pad=40, color=title_color)
plt.text(0.5, 1.05, subtitle, transform=ax.transAxes, ha='center', fontsize=13, fontweight='bold', color='#333333')
plt.text(0.5, 1.02, f"Initial Time: {base_time_str}", transform=ax.transAxes, ha='center', fontsize=11, color='#546e7a')

plt.savefig(output_png, bbox_inches='tight')
plt.close(fig)
print(f"Successfully generated {model_type} plot asset.")
EOF

    git add .
    git commit -m "Update plots"
    git push origin main
    echo "========================================================="
    
    # =========================================================
    # DYNAMIC SLEEP CALCULATION (Snaps to 02, 08, 14, 20 UTC)
    # =========================================================
    CURRENT_HOUR=$(date -u +%-H)
    CURRENT_MIN=$(date -u +%-M)
    CURRENT_SEC=$(date -u +%-S)

    # Find the next target hour
    if [ $CURRENT_HOUR -lt 2 ]; then NEXT_TARGET=2
    elif [ $CURRENT_HOUR -lt 8 ]; then NEXT_TARGET=8
    elif [ $CURRENT_HOUR -lt 14 ]; then NEXT_TARGET=14
    elif [ $CURRENT_HOUR -lt 20 ]; then NEXT_TARGET=20
    else NEXT_TARGET=26 # 26 hours means 02:00Z tomorrow
    fi

    HOURS_TO_WAIT=$((NEXT_TARGET - CURRENT_HOUR - 1))
    MINS_TO_WAIT=$((59 - CURRENT_MIN))
    SECS_TO_WAIT=$((60 - CURRENT_SEC))

    TOTAL_SLEEP=$(( (HOURS_TO_WAIT * 3600) + (MINS_TO_WAIT * 60) + SECS_TO_WAIT ))

    echo "Cycle completed. Snapping to next target interval ($((NEXT_TARGET % 24)):00Z)."
    echo "Sleeping for $TOTAL_SLEEP seconds..."
    echo "========================================================="
    
    sleep $TOTAL_SLEEP
done
#!/bin/bash

WORKSPACE="/Users/eknlau/VS_code/HKMETC-ensemble/"
cd "$WORKSPACE"

echo "========================================================="
echo " Starting Google Weather Lab Path-Optimized Engine"
echo "========================================================="

    python3.11 << 'EOF'
import os
import requests
import pandas as pd
from datetime import datetime, timedelta

# --- MATPLOTLIB HEADLESS RUNTIME ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

base_path = "/Users/eknlau/VS_code/GHMWS/ensemble-track/wp"

models = {
    "GENC": {
        "title": "GENC Ensemble Tracks - GHMWS"
    },
    "FNV3": {
        "title": "FNV3 Ensemble Tracks - GHMWS"
    }
}

bounds = [900, 915, 930, 945, 960, 970, 980, 990, 1000, 1010]
cmap = mcolors.ListedColormap(['#4a148c', '#880e4f', '#b71c1c', '#e65100', '#ff8f00', '#fbc02d', '#4db6ac', '#0277bd', '#808080'])
norm = mcolors.BoundaryNorm(bounds, cmap.N)

for model_name, cfg in models.items():
    print(f"[{datetime.now()}] Initializing search engine for {model_name}...")
    
    success = False
    for lookback_hours in range(0, 25, 6):
        now_utc = datetime.utcnow() - timedelta(hours=lookback_hours)
        
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
        
        yyyy = now_utc.strftime("%Y")
        mm = now_utc.strftime("%m")
        dd = now_utc.strftime("%d")
        date_folder = f"{yyyy}{mm}{dd}"
        cycle_str = f"{init_time:02d}Z"
        time_stamp_str = f"{yyyy}_{mm}_{dd}T{init_time:02d}_00"
        
        target_url = f"https://deepmind.google.com/science/weatherlab/download/cyclones/{model_name}/ensemble/cyclogenesis/csv/{model_name}_{time_stamp_str}_cyclogenesis.csv"
        
        print(f"--> Checking availability for {cycle_str} ({yyyy}-{mm}-{dd})...")
        
        try:
            r = requests.get(target_url, timeout=10)
            if r.status_code != 200:
                continue
                
            if r.text.lstrip().startswith("<!DOCTYPE html>") or "<html" in r.text.lower():
                continue
                
            path_2 = f"/Users/eknlau/VS_code/GHMWS/ensemble-track/wp/{model_name}/"
            run_dir = os.path.join(path_2, date_folder, cycle_str)
            os.makedirs(run_dir, exist_ok=True)
            
            local_csv_path = os.path.join(run_dir, f"{model_name.lower()}-unpaired-NWP.csv")
            archive_png = os.path.join(run_dir, "240.png")            
            latest_png = os.path.join(path_2, "240.png")
            
            with open(local_csv_path, 'wb') as f:
                f.write(r.content)
                
            # FIX 1: Add comment='#' to bypass Google's legal notice header lines
            df_2 = pd.read_csv(local_csv_path, comment='#')
            if df_2.empty:
                continue
                
            # FIX 2: Explicit map matching to your true file header keys
            if 'minimum_sea_level_pressure_hpa' in df_2.columns:
                df_2 = df_2.rename(columns={'minimum_sea_level_pressure_hpa': 'pressure'})
            
            # --- ADJUST NEGATIVE LONGITUDE HERE ---
            if 'lon' in df_2.columns:
                df_2.loc[df_2['lon'] < 0, 'lon'] += 360
            
            # Use 'valid_time' for perfect chronological track line connection sorting
            df_2 = df_2.sort_values(by=['track_id', 'sample', 'valid_time'])
            
            base_time_str = f"{yyyy}-{mm}-{dd} {cycle_str}"

            # -------------------------------------------------------------
            # INTEGRATED VISUALIZATION CANVAS
            # -------------------------------------------------------------
            fig = plt.figure(figsize=(12, 9), dpi=100)
            ax_2 = plt.axes(projection=ccrs.PlateCarree())
            ax_2.set_extent([100, 180, 0, 60], crs=ccrs.PlateCarree())

            ax_2.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=2)
            ax_2.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.4, zorder=2)
            ax_2.add_feature(cfeature.LAND, facecolor='#f5f5f5', zorder=1)

            gl = ax_2.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False

            # FIX 3: Group by track_id and sample matching the precise columns
            for _, group in df_2.groupby(['track_id', 'sample']):
                ax_2.plot(
                    group['lon'], group['lat'], color='gray', linewidth=0.6, alpha=0.2, 
                    transform=ccrs.PlateCarree(), zorder=3
                )

            sc = ax_2.scatter(
                df_2['lon'], df_2['lat'], 
                edgecolors=cmap(norm(df_2['pressure'])), 
                facecolors='none', 
                s=15,             
                linewidths=1.2,    
                alpha=0.9, 
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
            
            print(f"--> [SUCCESS] Processed {cycle_str}! Visual maps written safely.")
            success = True
            break 
            
        except Exception as e:
            print(f"    Internal cycle error for {cycle_str}: {e}")
            continue
            
    if not success:
        print(f"[WARN] Failed to find any valid recent cycles for {model_name} in the last 24 hours.")
EOF

    echo "Pushing updates to production repository main branch..."
    git add .
    git commit -m "Automated Sync: Core track headers aligned to true WeatherLab output specs"
    git push origin main
    
    echo "========================================================="
    
    # =========================================================
    # DYNAMIC SLEEP CALCULATION (Snaps to 02, 08, 14, 20 UTC)
    # =========================================================
    CURRENT_HOUR=$(date -u +%-H)
    CURRENT_MIN=$(date -u +%-M)
    CURRENT_SEC=$(date -u +%-S)

    # Determine the next sequential target interval
    if [ $CURRENT_HOUR -lt 2 ]; then NEXT_TARGET=2
    elif [ $CURRENT_HOUR -lt 8 ]; then NEXT_TARGET=8
    elif [ $CURRENT_HOUR -lt 14 ]; then NEXT_TARGET=14
    elif [ $CURRENT_HOUR -lt 20 ]; then NEXT_TARGET=20
    else NEXT_TARGET=26 # 26 represents 02:00Z the next calendar day
    fi

    HOURS_TO_WAIT=$((NEXT_TARGET - CURRENT_HOUR - 1))
    MINS_TO_WAIT=$((59 - CURRENT_MIN))
    SECS_TO_WAIT=$((60 - CURRENT_SEC))

    TOTAL_SLEEP=$(( (HOURS_TO_WAIT * 3600) + (MINS_TO_WAIT * 60) + SECS_TO_WAIT ))

    echo "Cycle execution complete. Snapping to next target interval ($((NEXT_TARGET % 24)):00Z)."
    echo "Sleeping for $TOTAL_SLEEP seconds..."
    echo "========================================================="
    
    sleep $TOTAL_SLEEP
done
