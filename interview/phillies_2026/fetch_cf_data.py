"""
Fetch Center Fielder Statcast Data from BigQuery (or generate synthetic if unavailable)
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pybaseball import statcast, playerid_reverse_lookup, cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT = "hankstank"
DATASET = "mlb_2026_season"
TABLE = "statcast_pitches"

cache.enable()


def generate_synthetic_cf_data(year_range=(2025, 2026), n_players=None):
    """
    Generate realistic synthetic center fielder defensive data for demo.
    
    Creates plays with varying skill levels to show meaningful model differentiation.
    If n_players is None, uses all available skill_profiles.
    """
    logger.info(f"Generating synthetic CF data ({year_range[0]}-{year_range[1]})...")
    
    np.random.seed(42)
    
    # Comprehensive 2025-2026 MLB center fielders + call-ups who played CF
    # (catch_baseline_boost, route_efficiency, arm_strength) - each 0-1 scale
    # Includes starters, backups, late-season call-ups, and utility players
    skill_profiles = {
        # Elite tier
        "Mike Trout": (0.16, 0.93, 0.94),
        "Byron Buxton": (0.14, 0.90, 0.88),
        "Luis Robert Jr.": (0.13, 0.89, 0.87),
        
        # Above-average tier
        "Harrison Bader": (0.11, 0.85, 0.84),
        "Julio Rodríguez": (0.10, 0.86, 0.82),
        "Cody Bellinger": (0.09, 0.87, 0.81),
        "Randy Arozarena": (0.08, 0.84, 0.80),
        "Cedric Mullins": (0.07, 0.83, 0.79),
        "Brenton Doyle": (0.06, 0.84, 0.77),
        "Brandon Marsh": (0.05, 0.81, 0.76),
        
        # Average tier
        "Kyle Schwarber": (-0.08, 0.82, 0.78),
        "Oscar Colas": (0.03, 0.82, 0.74),
        "Starling Marte": (0.04, 0.83, 0.77),
        "George Springer": (0.02, 0.80, 0.75),
        "Christian Yelich": (-0.01, 0.81, 0.74),
        "Whit Merrifield": (-0.02, 0.80, 0.72),
        "Andrew Benintendi": (-0.03, 0.79, 0.71),
        
        # Below-average/backup tier
        "Jarren Duran": (-0.05, 0.81, 0.73),
        "Trent Grisham": (-0.02, 0.79, 0.75),
        "Robbie Grossman": (-0.10, 0.78, 0.71),
        "Lewis Brinson": (-0.08, 0.80, 0.72),
        "Mitch Garver": (-0.12, 0.77, 0.68),
        "Trayce Thompson": (-0.09, 0.78, 0.70),
        "Esteury Ruiz": (-0.04, 0.80, 0.73),
        "Khalil Lee": (-0.11, 0.76, 0.67),
        
        # Occasional CF (primarily other OF)
        "Juan Soto": (-0.07, 0.81, 0.69),
        "Aaron Judge": (-0.05, 0.77, 0.76),
        "Kyle Tucker": (0.04, 0.83, 0.78),
        "Bryce Harper": (-0.01, 0.80, 0.73),
        
        # Late-season call-ups / emergency CF
        "Brent Rooker": (-0.06, 0.79, 0.70),
        "DaRon Bland": (-0.14, 0.74, 0.64),
        "Joc Pederson": (-0.03, 0.81, 0.72),
        "Austin Hays": (0.02, 0.82, 0.74),
        "Sonny Gray": (-0.15, 0.72, 0.61),  # Pitcher emergency OF
    }
    
    # If n_players not specified, use all available
    if n_players is None:
        n_players = len(skill_profiles)
        logger.info(f"Using all {n_players} skill profiles")
    else:
        logger.info(f"Using {n_players} of {len(skill_profiles)} skill profiles")
    
    records = []
    dates = pd.date_range(start=f"{year_range[0]}-04-01", end=f"{year_range[1]}-09-30", freq="D").to_pydatetime()
    
    for cf_name in list(skill_profiles.keys())[:n_players]:
        catch_boost, route_eff, arm_str = skill_profiles[cf_name]
        
        # Varied play volume by player tier
        if cf_name in ["Mitch Garver", "Lewis Brinson", "Khalil Lee", "DaRon Bland", "Sonny Gray"]:
            n_plays = np.random.randint(15, 50)       # Emergency/minimal: ~15-50 plays
        elif cf_name in ["Jarren Duran", "Trent Grisham", "Trayce Thompson", "Oscar Colas", "Esteury Ruiz"]:
            n_plays = np.random.randint(60, 110)      # Backup/developing: ~60-110
        elif cf_name in ["Robbie Grossman", "Whit Merrifield", "Andrew Benintendi", "Christian Yelich", "George Springer"]:
            n_plays = np.random.randint(50, 100)      # Utility/veteran: ~50-100
        elif cf_name in ["Juan Soto", "Aaron Judge", "Kyle Tucker", "Bryce Harper", "Joc Pederson", "Austin Hays", "Brent Rooker"]:
            n_plays = np.random.randint(40, 90)       # Occasional CF: ~40-90
        else:
            n_plays = np.random.randint(110, 160)     # Regular starters: ~110-160 plays
        
        for _ in range(n_plays):
            # Play scenario
            game_date = pd.Timestamp(np.random.choice(dates))
            exit_velo = np.random.normal(95, 8)  # mph, realistic flyball velos
            launch_angle = np.random.normal(28, 8)  # degrees, typical flyballs
            
            # Physics-based hang time
            hang_time = (2 * exit_velo * np.sin(np.radians(launch_angle))) / 32.174
            hang_time = max(0.5, hang_time)  # Realistic bounds
            
            # Distance (Statcast formula approximation)
            hit_distance = (exit_velo ** 1.5 / 100) * (launch_angle + 25)
            hit_distance = np.clip(hit_distance, 50, 450)
            
            # Physics-based catch probability
            physics_catch_prob = 1 / (1 + np.exp(-(hang_time - 2.0) / 0.5))  # Logistic
            
            # Fielder skill modulation
            catch_prob = np.clip(physics_catch_prob + catch_boost, 0.1, 0.95)
            is_out = 1 if np.random.random() < catch_prob else 0
            
            # Route quality
            geodesic = hit_distance
            route_arc = geodesic / route_eff + np.random.normal(0, 5)  # With noise
            
            # Arm opportunity context
            arm_opportunity_ft = hit_distance
            if hit_distance < 100:
                arm_context = "shallow"
            elif hit_distance < 200:
                arm_context = "medium"
            elif hit_distance < 300:
                arm_context = "deep"
            else:
                arm_context = "very_deep"
            
            records.append({
                "game_date": game_date,
                "game_year": game_date.year,
                "fielder_name": cf_name,
                "exit_velocity": exit_velo,
                "launch_angle": launch_angle,
                "hang_time_sec": hang_time,
                "hit_distance_ft": hit_distance,
                "is_out": is_out,
                "geodesic_ft": geodesic,
                "route_arc_ft": route_arc,
                "arm_opportunity_ft": arm_opportunity_ft,
                "arm_context": arm_context,
                "catch_baseline_prob": physics_catch_prob,
            })
    
    df = pd.DataFrame(records)
    df['game_date'] = pd.to_datetime(df['game_date'])  # Ensure datetime type
    logger.info(f"Generated {len(df)} synthetic CF plays across {n_players} players")
    return df


def generate_synthetic_cf_data_from_roster(year_range=(2025, 2026), cf_roster=None):
    """
    Generate realistic synthetic center fielder defensive data using actual CF roster.
    
    For players not in our predefined skill_profiles, generates realistic but synthetic skills.
    """
    logger.info(f"Generating synthetic CF data from roster ({len(cf_roster)} players)...")
    
    np.random.seed(42)
    
    # Predefined profiles for known players
    known_profiles = {
        # Elite tier
        "Mike Trout": (0.16, 0.93, 0.94),
        "Byron Buxton": (0.14, 0.90, 0.88),
        "Luis Robert Jr.": (0.13, 0.89, 0.87),
        "Julio Rodríguez": (0.10, 0.86, 0.82),
        
        # Above-average tier
        "Harrison Bader": (0.11, 0.85, 0.84),
        "Cody Bellinger": (0.09, 0.87, 0.81),
        "Randy Arozarena": (0.08, 0.84, 0.80),
        "Cedric Mullins": (0.07, 0.83, 0.79),
        "Brenton Doyle": (0.06, 0.84, 0.77),
        "Brandon Marsh": (0.05, 0.81, 0.76),
        
        # Average tier
        "Kyle Schwarber": (-0.08, 0.82, 0.78),
        "Oscar Colas": (0.03, 0.82, 0.74),
        "Starling Marte": (0.04, 0.83, 0.77),
        "George Springer": (0.02, 0.80, 0.75),
        "Christian Yelich": (-0.01, 0.81, 0.74),
        "Whit Merrifield": (-0.02, 0.80, 0.72),
        "Andrew Benintendi": (-0.03, 0.79, 0.71),
        
        # Below-average/backup tier
        "Jarren Duran": (-0.05, 0.81, 0.73),
        "Trent Grisham": (-0.02, 0.79, 0.75),
        "Robbie Grossman": (-0.10, 0.78, 0.71),
        "Lewis Brinson": (-0.08, 0.80, 0.72),
        "Mitch Garver": (-0.12, 0.77, 0.68),
        "Trayce Thompson": (-0.09, 0.78, 0.70),
        "Esteury Ruiz": (-0.04, 0.80, 0.73),
        "Khalil Lee": (-0.11, 0.76, 0.67),
        "Juan Soto": (-0.07, 0.81, 0.69),
        "Aaron Judge": (-0.05, 0.77, 0.76),
        "Kyle Tucker": (0.04, 0.83, 0.78),
        "Bryce Harper": (-0.01, 0.80, 0.73),
        "Brent Rooker": (-0.06, 0.79, 0.70),
        "Joc Pederson": (-0.03, 0.81, 0.72),
        "Austin Hays": (0.02, 0.82, 0.74),
    }
    
    records = []
    dates = pd.date_range(start=f"{year_range[0]}-04-01", end=f"{year_range[1]}-09-30", freq="D").to_pydatetime()
    
    for idx, cf_name in enumerate(cf_roster):
        # Get profile if known, otherwise generate realistic one
        if cf_name in known_profiles:
            catch_boost, route_eff, arm_str = known_profiles[cf_name]
        else:
            # Generate realistic skill profile for unknown players
            # Use position in roster as heuristic for playing time
            percentile = idx / len(cf_roster)
            catch_boost = np.random.normal(-0.02, 0.08) * (1 - percentile * 0.5)
            route_eff = 0.75 + np.random.normal(0, 0.05)
            arm_str = 0.70 + np.random.normal(0, 0.06)
        
        # Vary play volume based on appearance rank (top = starters, bottom = emergency CF)
        appearances_percentile = idx / len(cf_roster)
        if appearances_percentile < 0.15:  # Top 15% - regular starters
            n_plays = np.random.randint(110, 160)
        elif appearances_percentile < 0.40:  # Next 25% - regular starters
            n_plays = np.random.randint(100, 150)
        elif appearances_percentile < 0.65:  # Middle 25% - backup/regular
            n_plays = np.random.randint(60, 110)
        elif appearances_percentile < 0.90:  # Next 25% - occasional
            n_plays = np.random.randint(30, 70)
        else:  # Bottom 10% - emergency CF
            n_plays = np.random.randint(10, 40)
        
        for _ in range(n_plays):
            # Play scenario
            game_date = pd.Timestamp(np.random.choice(dates))
            exit_velo = np.random.normal(95, 8)  # mph, realistic flyball velos
            launch_angle = np.random.normal(28, 8)  # degrees, typical flyballs
            
            # Physics-based hang time
            hang_time = (2 * exit_velo * np.sin(np.radians(launch_angle))) / 32.174
            hang_time = max(0.5, hang_time)  # Realistic bounds
            
            # Distance (Statcast formula approximation)
            hit_distance = (exit_velo ** 1.5 / 100) * (launch_angle + 25)
            hit_distance = np.clip(hit_distance, 50, 450)
            
            # Physics-based catch probability
            physics_catch_prob = 1 / (1 + np.exp(-(hang_time - 2.0) / 0.5))  # Logistic
            
            # Fielder skill modulation
            catch_prob = np.clip(physics_catch_prob + catch_boost, 0.1, 0.95)
            is_out = 1 if np.random.random() < catch_prob else 0
            
            # Route quality
            geodesic = hit_distance
            route_arc = geodesic / route_eff + np.random.normal(0, 5)  # With noise
            
            # Arm opportunity context
            arm_opportunity_ft = hit_distance
            if hit_distance < 100:
                arm_context = "shallow"
            elif hit_distance < 200:
                arm_context = "medium"
            elif hit_distance < 300:
                arm_context = "deep"
            else:
                arm_context = "very_deep"
            
            records.append({
                "game_date": game_date,
                "game_year": game_date.year,
                "fielder_name": cf_name,
                "exit_velocity": exit_velo,
                "launch_angle": launch_angle,
                "hang_time_sec": hang_time,
                "hit_distance_ft": hit_distance,
                "is_out": is_out,
                "geodesic_ft": geodesic,
                "route_arc_ft": route_arc,
                "arm_opportunity_ft": arm_opportunity_ft,
                "arm_context": arm_context,
                "catch_baseline_prob": physics_catch_prob,
            })
    
    df = pd.DataFrame(records)
    df['game_date'] = pd.to_datetime(df['game_date'])
    logger.info(f"Generated {len(df)} synthetic CF plays across {len(cf_roster)} players")
    return df


OUT_EVENTS = {
    'field_out',
    'force_out',
    'triple_play',
    'double_play',
    'fielders_choice_out',
    'sac_fly',
    'sac_fly_double_play',
}


def _date_chunks(start_date: str, end_date: str, chunk_days: int = 14):
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        yield current.isoformat(), chunk_end.isoformat()
        current = chunk_end + timedelta(days=1)


def _lookup_cf_names(cf_ids: np.ndarray) -> dict:
    if len(cf_ids) == 0:
        return {}

    lookup = playerid_reverse_lookup([int(x) for x in cf_ids], key_type='mlbam')
    if lookup is None or lookup.empty:
        return {int(x): f"mlbam_{int(x)}" for x in cf_ids}

    id_to_name = {}
    for _, row in lookup.iterrows():
        try:
            mlbam = int(row['key_mlbam'])
        except Exception:
            continue
        first = str(row.get('name_first', '')).strip().title()
        last = str(row.get('name_last', '')).strip().title()
        full_name = f"{first} {last}".strip()
        id_to_name[mlbam] = full_name if full_name else f"mlbam_{mlbam}"

    for cf_id in cf_ids:
        cid = int(cf_id)
        if cid not in id_to_name:
            id_to_name[cid] = f"mlbam_{cid}"

    return id_to_name


def fetch_cf_statcast_data(year_range=(2025, 2026)):
    """
    REAL-DATA ONLY fetch.

    Pull Statcast from pybaseball and attribute batted-ball events to the center
    fielder on the play using `fielder_8` (MLBAM id for CF position).
    """
    start_date = f"{year_range[0]}-04-01"
    end_date = f"{year_range[1]}-09-30"
    logger.info(f"Fetching REAL Statcast CF data ({start_date} -> {end_date})")

    chunks = []
    for chunk_start, chunk_end in _date_chunks(start_date, end_date, chunk_days=14):
        logger.info(f"  Pulling Statcast chunk {chunk_start} to {chunk_end}")
        df_chunk = statcast(start_dt=chunk_start, end_dt=chunk_end)
        if df_chunk is not None and len(df_chunk) > 0:
            chunks.append(df_chunk)

    if not chunks:
        raise RuntimeError("No Statcast data returned for requested date range")

    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"Fetched {len(df)} total Statcast pitches")

    # Real batted-ball events only
    df = df[df['type'] == 'X'].copy()
    df = df[df['bb_type'].isin(['fly_ball', 'line_drive', 'popup'])].copy()
    df = df[df['hit_distance_sc'].notna() & (df['hit_distance_sc'] > 50)].copy()
    df = df[df['fielder_8'].notna()].copy()

    if len(df) == 0:
        raise RuntimeError("No qualifying CF plays found in real Statcast data")

    # Map CF IDs -> names
    df['fielder_8'] = df['fielder_8'].astype(int)
    id_to_name = _lookup_cf_names(df['fielder_8'].unique())
    df['fielder_name'] = df['fielder_8'].map(id_to_name)

    # Build downstream-compatible columns from real values
    df['game_date'] = pd.to_datetime(df['game_date'])
    df['game_year'] = df['game_date'].dt.year
    df['exit_velocity'] = df['launch_speed'].fillna(df['launch_speed'].median())
    df['launch_angle'] = df['launch_angle'].fillna(df['launch_angle'].median())
    df['hit_distance_ft'] = df['hit_distance_sc']

    hang_time = (2 * df['exit_velocity'] * np.sin(np.radians(df['launch_angle']))) / 32.174
    df['hang_time_sec'] = np.clip(hang_time, 0.5, 8.0)

    df['is_out'] = df['events'].isin(OUT_EVENTS).astype(int)
    df['geodesic_ft'] = df['hit_distance_ft']

    if 'hc_x' in df.columns and df['hc_x'].notna().any():
        lateral = (df['hc_x'].fillna(125.0) - 125.0).abs() / 125.0
    else:
        lateral = 0.0

    if 'of_fielding_alignment' in df.columns and df['of_fielding_alignment'].notna().any():
        depth_factor = df['of_fielding_alignment'].fillna('Standard').map({
            'Strategic': 0.05,
            '4th outfielder': 0.04,
            'Standard': 0.02,
        }).fillna(0.02)
    else:
        depth_factor = 0.02

    df['route_arc_ft'] = df['geodesic_ft'] * (1.02 + 0.10 * lateral + depth_factor)
    df['arm_opportunity_ft'] = df['hit_distance_ft']
    df['arm_context'] = pd.cut(
        df['hit_distance_ft'],
        bins=[0, 100, 200, 300, np.inf],
        labels=['shallow', 'medium', 'deep', 'very_deep'],
    ).astype(str)
    df['catch_baseline_prob'] = 1 / (1 + np.exp(-(df['hang_time_sec'] - 2.0) / 0.5))

    output = df[[
        'game_date', 'game_year', 'fielder_name', 'exit_velocity', 'launch_angle',
        'hang_time_sec', 'hit_distance_ft', 'is_out', 'geodesic_ft', 'route_arc_ft',
        'arm_opportunity_ft', 'arm_context', 'catch_baseline_prob'
    ]].copy()

    logger.info(f"Built REAL CF dataset: {len(output)} plays across {output['fielder_name'].nunique()} CF")
    return output


def summary_stats(df):
    """Log basic summary statistics about the data"""
    logger.info(f"=== CF STATCAST DATA SUMMARY ===")
    logger.info(f"Total plays: {len(df)}")
    logger.info(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    logger.info(f"Unique players: {df['fielder_name'].nunique()}")
    logger.info(f"Total fielding outs: {df['is_out'].sum()}")
    logger.info(f"Overall out rate: {df['is_out'].mean():.2%}")
    if 'exit_velocity' in df.columns and df['exit_velocity'].notna().sum() > 0:
        logger.info(f"Avg exit velocity: {df['exit_velocity'].mean():.1f} mph")
    if 'launch_angle' in df.columns and df['launch_angle'].notna().sum() > 0:
        logger.info(f"Avg launch angle: {df['launch_angle'].mean():.1f}°")
    if 'hang_time_sec' in df.columns and df['hang_time_sec'].notna().sum() > 0:
        logger.info(f"Avg hang time: {df['hang_time_sec'].mean():.2f} sec")
    if 'hit_distance_ft' in df.columns and df['hit_distance_ft'].notna().sum() > 0:
        logger.info(f"Avg hit distance: {df['hit_distance_ft'].mean():.1f} ft")
    

if __name__ == "__main__":
    df = fetch_cf_statcast_data()
    summary_stats(df)
    df.to_parquet("cf_statcast_raw.parquet", index=False)
    logger.info("Saved to cf_statcast_raw.parquet")
