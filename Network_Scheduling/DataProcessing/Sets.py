import pandas as pd
from pathlib import Path
import os  # For mkdir

# Set base directory
base_dir = Path(__file__).parent.parent

# Print control flags
PRINT_TRADES = False
PRINT_PORTS = True
PRINT_COMMITMENTS = True
PRINT_LANES = False
PRINT_TSP = False
PRINT_RELEVANT_PORTS = False
PRINT_SUMMARY = False

def normalize_port_name(port):
    """
    Standardize port name: remove commas, normalize spaces, return cleaned.
    E.g., 'Freeport, Tx' → 'Freeport Tx', 'Wilmington, De' → 'Wilmington De'.
    """
    if pd.isna(port):
        return ''
    original = str(port).strip()
    cleaned = original.replace(',', '').replace('  ', ' ').strip()
    return cleaned

# ================================
# SERVICE CORRIDORS (TRADES)
# ================================
def load_trades():
    """
    Load service corridor (trade) information from Trade_Constraints.csv and Transit_Times.csv
    Returns a dictionary where key = trade code, value = dict of all attributes
    """
    trade_path = base_dir / 'DATA/Master/Trade_Constraints/Trade_Constraints.csv'
    df = pd.read_csv(trade_path)
    
    # Apply same logic as Analysis.py: strip column names and clean data
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=[df.columns[0]])
    
    # Convert to dictionary with trade code as key
    trades = {}
    for _, row in df.iterrows():
        trade_code = row[df.columns[0]].strip()
        trades[trade_code] = {
            'name': row['Name'].strip() if pd.notna(row['Name']) else '',
            'min_spread': int(row['Min spread']) if pd.notna(row['Min spread']) else 0,
            'min_loading_ports': int(row['Min Loading Ports']) if pd.notna(row['Min Loading Ports']) else 0,
            'max_loading_ports': int(row['Max Loading Ports']) if pd.notna(row['Max Loading Ports']) else 0,
            'min_discharge_ports': int(row['Min Discharge Ports']) if pd.notna(row['Min Discharge Ports']) else 0,
            'max_discharge_ports': int(row['Max Discharge Ports']) if pd.notna(row['Max Discharge Ports']) else 0,
            'route_start_range_lower': int(row['Route start range lower']) if pd.notna(row['Route start range lower']) else 0,
            'route_start_range_upper': int(row['Route start range upper']) if pd.notna(row['Route start range upper']) else 0,
            'route_dead_position_lower': int(row['Route dead position lower']) if pd.notna(row['Route dead position lower']) else 0,
            'route_dead_position_upper': int(row['Route dead position upper']) if pd.notna(row['Route dead position upper']) else 0,
            'transit_times': {}  # Will be populated from Transit_Times.csv
        }
    
    # Load transit times from Transit_Times.csv
    transit_path = base_dir / 'DATA/Master/Port_Constraints/Transit_Times.csv'
    df_transit = pd.read_csv(transit_path)
    
    # Apply same logic: strip column names and clean data
    df_transit.columns = df_transit.columns.str.strip()
    df_transit = df_transit.dropna(subset=[df_transit.columns[0]])
    
    trade_col = df_transit.columns[0]
    port_from_col = df_transit.columns[1]
    port_to_col = df_transit.columns[2]
    time_col = df_transit.columns[3]  # The "-" column contains transit times
    
    # Add transit times to the appropriate trades
    for _, row in df_transit.iterrows():
        trade = row[trade_col].strip()
        port_from = row[port_from_col].strip()
        port_to = row[port_to_col].strip()
        transit_time = int(row[time_col]) if pd.notna(row[time_col]) else 0
        
        if trade in trades:
            # Store as nested dict with (port_from, port_to) tuple as key
            trades[trade]['transit_times'][(port_from, port_to)] = transit_time
        else:
            print(f"Warning: Trade '{trade}' from Transit_Times.csv not found in Trade_Constraints.csv")
    
    return trades


# ================================
# PORTS
# ================================
def load_ports():
    """
    Load port information from Port_Information.csv and Port_Constraints.csv
    Returns a dictionary where key = port name, value = dict of all attributes
    """
    port_path = base_dir / 'DATA/Master/Port_Constraints/Port_Information.csv'
    df = pd.read_csv(port_path)
    
    # Apply same logic as Analysis.py: strip column names and clean data
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=[df.columns[0]])
    
    # Convert to dictionary with port name as key
    ports = {}
    for _, row in df.iterrows():
        port_name = row[df.columns[0]].strip()
        ports[port_name] = {
            'area': row['Area'].strip() if pd.notna(row['Area']) else '',
            'sub_area': row['Sub-area'].strip() if pd.notna(row['Sub-area']) else '',
            'call_duration_days': float(row['🟢 Call duration [days]']) if pd.notna(row['🟢 Call duration [days]']) else 0.0,
            'port_delays_days': float(row['🟢 Port delays [days]']) if pd.notna(row['🟢 Port delays [days]']) else 0.0,
            'total_port_time_days': float(row['Total port time[days]']) if pd.notna(row['Total port time[days]']) else 0.0,
            'port_type': row['Port type'].strip() if pd.notna(row['Port type']) else '',
            'in_eu': row['In EU'].strip() if pd.notna(row['In EU']) else '',
            'waypoint': int(row['Waypoint']) if pd.notna(row['Waypoint']) else 0,
            'area_point': int(row['Area point']) if pd.notna(row['Area point']) else 0,
            'connection_point': int(row['Connection point']) if pd.notna(row['Connection point']) else 0,
            'longitude': float(row['Longitude']) if pd.notna(row['Longitude']) else 0.0,
            'latitude': float(row['Latitude']) if pd.notna(row['Latitude']) else 0.0,
            'trade_constraints': {}  # Will be populated from Port_Constraints.csv
        }
    
    # Load port constraints from Port_Constraints.csv
    constraints_path = base_dir / 'DATA/Master/Port_Constraints/Port_Constraints.csv'
    df_constraints = pd.read_csv(constraints_path)
    
    # Apply same logic: strip column names and clean data
    df_constraints.columns = df_constraints.columns.str.strip()
    df_constraints = df_constraints.dropna(subset=[df_constraints.columns[0]])
    
    trade_col = df_constraints.columns[0]
    port_col = df_constraints.columns[1]
    min_spread_col = df_constraints.columns[2]
    pos_lower_col = df_constraints.columns[3]
    pos_upper_col = df_constraints.columns[4]
    usin_col = df_constraints.columns[5]
    
    # Map port name variations (handle cases like "Charleston, SC" vs "CharlestonSC")
    port_name_map = {}
    for port_name in ports.keys():
        # Create a normalized version without spaces and commas
        normalized = port_name.replace(',', '').replace(' ', '')
        port_name_map[normalized] = port_name
        # Also keep the original
        port_name_map[port_name] = port_name
    
    # Add constraints to the appropriate ports
    for _, row in df_constraints.iterrows():
        trade = row[trade_col].strip()
        port_name_raw = row[port_col].strip()
        
        # Try to find the port in our ports dictionary
        # First try exact match, then try normalized version
        port_name = None
        if port_name_raw in ports:
            port_name = port_name_raw
        else:
            # Try normalized matching
            normalized = port_name_raw.replace(',', '').replace(' ', '')
            if normalized in port_name_map:
                port_name = port_name_map[normalized]
        
        if port_name and port_name in ports:
            ports[port_name]['trade_constraints'][trade] = {
                'min_spread': int(row[min_spread_col]) if pd.notna(row[min_spread_col]) else 0,
                'desired_position_lower': int(row[pos_lower_col]) if pd.notna(row[pos_lower_col]) else 0,
                'desired_position_upper': int(row[pos_upper_col]) if pd.notna(row[pos_upper_col]) else 0,
                'port_on_usin': int(row[usin_col]) if pd.notna(row[usin_col]) else 0
            }
        else:
            print(f"Warning: Port '{port_name_raw}' from Port_Constraints.csv not found in Port_Information.csv")
    
    return ports


# ================================
# COMMITMENTS
# ================================
def load_commitments():
    """Load commitments from Commitments_Prio.csv as {trade_code: {(from_port, to_port): frequency}}"""
    base_dir = Path(__file__).parent.parent
    commitment_path = base_dir / "DATA/Master/Commitments/Commitments_Prio.csv"
    
    if not commitment_path.exists():
        print(f"Warning: {commitment_path} not found, returning empty commitments")
        return {}
    
    # Read raw lines to handle malformed CSV (trailing commas, extra fields)
    with open(commitment_path, 'r') as f:
        lines = f.readlines()
    
    # Clean lines: split by comma, clean each field (strip ws, remove inline comments), take first 4
    cleaned_lines = []
    for line_num, line in enumerate(lines, 1):
        fields = [f.strip().split('#')[0].strip() for f in line.strip().split(',')]  # Strip ws and comments per field
        if len(fields) > 4:
            cleaned = ','.join(fields[:4])
            #print(f"Cleaned line {line_num}: truncated {len(fields)} fields to 4")
        else:
            cleaned = ','.join(fields)
        cleaned_lines.append(cleaned)
    
    # Write cleaned to temp string and read with pandas
    from io import StringIO
    cleaned_csv = '\n'.join(cleaned_lines)
    df = pd.read_csv(StringIO(cleaned_csv))
    
    # Normalize port names and build dict
    commitments = {}
    for _, row in df.iterrows():
        trade_code = normalize_port_name(str(row.iloc[0]).strip())
        from_port = normalize_port_name(str(row.iloc[1]).strip())
        to_port = normalize_port_name(str(row.iloc[2]).strip())
        freq_str = str(row.iloc[3]).strip().split('#')[0].strip()  # Extra strip for safety
        frequency = int(freq_str) if pd.notna(row.iloc[3]) and freq_str else 0
        
        if trade_code not in commitments:
            commitments[trade_code] = {}
        commitments[trade_code][(from_port, to_port)] = frequency
    
    #print(f"Loaded {len(commitments)} trades with commitments from {commitment_path}")
    return commitments


# ================================
# DISTANCES (Global)
# ================================
def load_distances():
    """
    Load global port-to-port distances from Distances.csv.
    Returns a dictionary: {(port_from, port_to): distance_nm}
    Skips self-distances (0); mirrors for symmetry (adds (B,A) if only (A,B) present).
    Use 'Distance [nm]' for primary sailing distances.
    """
    distance_path = base_dir / 'DATA/Master/Distances/Distances.csv'
    df = pd.read_csv(distance_path)
    
    # Strip columns and clean data
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['Port from', 'Port to'])
    
    distances = {}
    for _, row in df.iterrows():
        port_from = row['Port from'].strip()
        port_to = row['Port to'].strip()
        distance_nm = float(row['Distance [nm]']) if pd.notna(row['Distance [nm]']) else 0.0
        
        if distance_nm > 0:  # Skip self-distances
            distances[(port_from, port_to)] = distance_nm
    
    # Mirror for symmetry (add reverse if not present)
    mirrored = {}
    for (p1, p2), dist in list(distances.items()):
        if (p2, p1) not in distances:
            mirrored[(p2, p1)] = dist
    distances.update(mirrored)
    
    return distances


# ================================
# REQUESTED LANES (Global)
# ================================
def load_requested_lanes():
    """
    Load requested lane information from Requested_Lanes.csv.
    Defines lane variants per trade with requested positions (vessels), USIN flag, and regional sequences.
    
    Returns: {trade_code: [list of lane dicts]}
    Each lane: {'name': str, 'positions': int, 'is_usin': bool, 'regions': [str]}  # Non-empty region codes
    """
    lanes_path = base_dir / 'DATA/Master/Trade_Constraints/Requested_Lanes.csv'
    df = pd.read_csv(lanes_path)
    
    # Strip columns and clean data
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['Trade'])  # Drop rows with missing trade
    
    requested_lanes = {}
    region_cols = [col for col in df.columns if col in [str(i) for i in range(1,8)]]  # Cols '1','2',...,'7'
    
    for _, row in df.iterrows():
        trade_code = row['Trade'].strip()
        lane_name = row['Trade lane'].strip()
        positions = int(row['# Requested positions']) if pd.notna(row['# Requested positions']) else 0
        is_usin = bool(int(row['Is USIN'])) if pd.notna(row['Is USIN']) else False
        
        # Extract non-empty regions
        regions = []
        for col in region_cols:
            region = row[col].strip() if pd.notna(row[col]) else ''
            if region:  # Skip empties
                regions.append(region)
        
        lane_data = {
            'name': lane_name,
            'positions': positions,
            'is_usin': is_usin,
            'regions': regions
        }
        
        if trade_code not in requested_lanes:
            requested_lanes[trade_code] = []
        requested_lanes[trade_code].append(lane_data)
    
    return requested_lanes


def get_relevant_ports_from_commitments(commitments, trade_code):
    """
    Returns set of unique ports (from/to) from commitments for a specific trade.
    Used to filter TSP sequences to relevant ports only.
    
    Args:
        commitments: from load_commitments()
        trade_code: e.g., 'EUAF'
    
    Returns:
        set of str (port names); empty if no commitments
    """
    if trade_code not in commitments or not commitments[trade_code]:
        return set()
    
    ports = set()
    for (port_from, port_to), _ in commitments[trade_code].items():
        ports.add(port_from)
        ports.add(port_to)
    
    return ports


def load_and_filter_tsp(tsp_path, relevant_ports):
    """
    Loads and filters a TSP sequence CSV to only relevant ports.
    Filters rows where 'Port' in relevant_ports, preserves sequence order.
    
    Args:
        tsp_path: Path to TSP CSV (e.g., PortSequence_EUAFviaCapetoOC.csv)
        relevant_ports: set of str from get_relevant_ports_from_commitments
    
    Returns:
        List of dicts: [{'port': str, 'sequence': int, 'sub_area': str, 'sailing_nm': float, 'sailing_days': float}]
        Empty list if no file or no matching ports; None if file missing.
    """
    if not os.path.exists(tsp_path):
        return None
    
    # Read as text to clean lines (existing logic for trailing commas)
    with open(tsp_path, 'r') as f:
        lines = f.readlines()
    
    cleaned_lines = []
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        fields = [f.strip() for f in line.split(',')[:5]]
        if len(fields) < 5:
            print(f"Warning: Line {i} in {tsp_path.name} has <5 fields, skipping: {line}")
            continue
        if len([f for f in line.split(',')]) > 5:
            print(f"Warning: Cleaned line {i} in {tsp_path.name}: trailing fields ignored")
            clean_line = ','.join(fields)
            cleaned_lines.append(clean_line)
        else:
            cleaned_lines.append(line)
    
    if not cleaned_lines:
        return []
    
    from io import StringIO
    df = pd.read_csv(StringIO('\n'.join(cleaned_lines)), header=None, names=['Port', 'Sequence', 'Sub-area', 'Sailing [nm]', 'Sailing [day]'])
    df = df.dropna(subset=['Port'])
    
    # Normalize Port for filtering
    df['Port'] = df['Port'].apply(normalize_port_name)
    
    filtered_df = df[df['Port'].isin(relevant_ports)].copy()
    if filtered_df.empty:
        return []
    
    tsp_data = []
    for _, row in filtered_df.iterrows():
        tsp_data.append({
            'port': row['Port'].strip(),  # Normalized
            'sequence': int(row['Sequence']) if pd.notna(row['Sequence']) else 0,
            'sub_area': row['Sub-area'].strip() if pd.notna(row['Sub-area']) else '',
            'sailing_nm': float(row['Sailing [nm]']) if pd.notna(row['Sailing [nm]']) else 0.0,
            'sailing_days': float(row['Sailing [day]']) if pd.notna(row['Sailing [day]']) else 0.0
        })
    
    tsp_data.sort(key=lambda x: x['sequence'])
    return tsp_data

NewTSP = load_and_filter_tsp


def load_optimal_tsp(tsp_path):
    """
    Loads an optimal TSP sequence CSV (from the optimal TSP solver).
    These sequences are already optimal and don't need filtering.
    
    Args:
        tsp_path: Path to optimal TSP CSV (e.g., optimal_subset_sequence_EUAFviaCapetoOC.csv)
    
    Returns:
        List of dicts: [{'port': str, 'sequence': int, 'sub_area': str}]
        Empty list if no file; None if file missing.
    """
    if not os.path.exists(tsp_path):
        return None
    
    try:
        df = pd.read_csv(tsp_path)
        
        # Handle different column name formats
        port_col = None
        seq_col = None
        sub_area_col = None
        
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in ['port', 'ports']:
                port_col = col
            elif col_lower in ['sequence', 'seq']:
                seq_col = col
            elif col_lower in ['sub-area', 'sub_area', 'sub area']:
                sub_area_col = col
        
        if port_col is None or seq_col is None:
            print(f"Warning: Could not find Port and Sequence columns in {tsp_path.name}")
            return []
        
        df = df.dropna(subset=[port_col])
        
        # Normalize port names
        df[port_col] = df[port_col].apply(normalize_port_name)
        
        tsp_data = []
        for _, row in df.iterrows():
            tsp_data.append({
                'port': str(row[port_col]).strip(),
                'sequence': int(row[seq_col]) if pd.notna(row[seq_col]) else 0,
                'sub_area': str(row[sub_area_col]).strip() if sub_area_col and pd.notna(row.get(sub_area_col)) else '',
                'sailing_nm': 0.0,  # Not in optimal TSP format
                'sailing_days': 0.0  # Not in optimal TSP format
            })
        
        tsp_data.sort(key=lambda x: x['sequence'])
        return tsp_data
    except Exception as e:
        print(f"Error loading optimal TSP from {tsp_path}: {e}")
        return None


def build_distance_matrix_for_lane(filtered_tsp, distances, ports):
    """
    Build a distance matrix for a filtered TSP sequence, respecting DAG constraints.
    Creates all possible arcs (port_i -> port_j) where sequence_i < sequence_j.
    
    Args:
        filtered_tsp: List of dicts from load_and_filter_tsp (filtered ports in sequence order)
        distances: Global distances dict from load_distances()
        ports: Ports dict from load_ports() (to get port durations)
    
    Returns:
        Dict {(port_i, port_j): {
            'distance_nm': float,
            'port_time_days': float,  # Total port time at port_i
            'sailing_time_days': float,  # Time to sail from i to j
            'total_arc_time_days': float  # port_time + sailing_time
        }} for all valid DAG arcs.
        Missing distances logged as warnings.
    
    Speed rules:
        - 14 knots for distances < 400 nm (coastal/short hops)
        - 15.5 knots for distances >= 400 nm (open ocean)
    """
    if not filtered_tsp:
        return {}
    
    distance_matrix = {}
    n = len(filtered_tsp)
    
    for i in range(n):
        port_i = filtered_tsp[i]['port']
        seq_i = filtered_tsp[i]['sequence']
        
        # Get port duration at port_i
        port_time_i = ports[port_i]['total_port_time_days'] if port_i in ports else 0.0
        
        for j in range(i + 1, n):  # Only forward arcs (DAG)
            port_j = filtered_tsp[j]['port']
            seq_j = filtered_tsp[j]['sequence']
            
            # Sanity check: sequence should be increasing
            if seq_j <= seq_i:
                print(f"Warning: Sequence violation in TSP: {port_i}(seq{seq_i}) -> {port_j}(seq{seq_j})")
                continue
            
            # Lookup distance
            dist_nm = None
            if (port_i, port_j) in distances:
                dist_nm = distances[(port_i, port_j)]
            elif (port_j, port_i) in distances:
                dist_nm = distances[(port_j, port_i)]
            else:
                print(f"Warning: Distance not found for {port_i} -> {port_j} (skipping arc)")
                continue
            
            # Calculate sailing time based on distance
            if dist_nm < 400:
                speed_knots = 14.0  # Coastal/short sailing
            else:
                speed_knots = 15.5  # Open ocean
            
            # Sailing time = distance / (speed * 24 hours)
            sailing_time_days = dist_nm / (speed_knots * 24)
            
            # Total arc time = port time at origin + sailing time
            total_arc_time = port_time_i + sailing_time_days
            
            distance_matrix[(port_i, port_j)] = {
                'distance_nm': dist_nm,
                'port_time_days': port_time_i,
                'sailing_time_days': sailing_time_days,
                'total_arc_time_days': total_arc_time
            }
    
    return distance_matrix






# ================================
# HELPER FUNCTIONS FOR MIP ACCESS
# ================================

def get_trade_list(trades):
    """
    Returns a list of all trade codes.
    Useful for looping through trades to solve MIPs sequentially.
    
    Example:
        for trade_code in get_trade_list(trades):
            solve_mip_for_trade(trade_code)
    """
    return list(trades.keys())


def get_all_data_for_trade(trades, ports, commitments, requested_lanes, distances, trade_code):
    """
    Returns ALL relevant data for a specific trade in one consolidated structure.
    Now includes filtered TSP sequences per lane for DAG routing in MIPs.
    
    Args:
        trades: from load_trades()
        ports: from load_ports()
        commitments: from load_commitments()
        requested_lanes: from load_requested_lanes()
        distances: from load_distances()
        trade_code: e.g., 'EUAF'
    
    Returns:
        Same as before, plus:
        'tsp_sequences': {lane_name: filtered_tsp_list}  # Per-lane filtered sequences
        'distance_matrices': {lane_name: {(port_i, port_j): distance_nm}}  # DAG arc costs
    """
    if trade_code not in trades:
        return None
    
    trade_data = trades[trade_code]
    
    # Existing route start/dead (unchanged)
    route_start = None
    if trade_data['route_start_range_lower'] > 0 or trade_data['route_start_range_upper'] > 0:
        route_start = {
            'lower': trade_data['route_start_range_lower'],
            'upper': trade_data['route_start_range_upper']
        }
    
    route_dead = None
    if trade_data['route_dead_position_lower'] > 0 or trade_data['route_dead_position_upper'] > 0:
        route_dead = {
            'lower': trade_data['route_dead_position_lower'],
            'upper': trade_data['route_dead_position_upper']
        }
    
    # Existing ports_with_constraints (unchanged)
    ports_with_constraints = get_port_constraints_for_trade(ports, trade_code)
    
    # Existing commitments (unchanged)
    trade_commitments = commitments.get(trade_code, {}) if commitments else {}

    # New: Classify ports as loading, discharging, co-loading based on commitments
    loading_ports = set()
    discharging_ports = set()
    for (from_port, to_port), freq in trade_commitments.items():
        loading_ports.add(from_port)
        discharging_ports.add(to_port)
    co_loading_ports = loading_ports.intersection(discharging_ports)

    # Add to return data (as lists for easy iteration in MIP)
    port_classifications = {
        'loading_ports': sorted(list(loading_ports)),
        'discharging_ports': sorted(list(discharging_ports)),
        'co_loading_ports': sorted(list(co_loading_ports))
    }
    
    # Existing requested_lanes (unchanged)
    trade_lanes = requested_lanes.get(trade_code, []) if requested_lanes else []
    total_positions = sum(lane['positions'] for lane in trade_lanes)
    num_lanes = len(trade_lanes)
    requested_lanes_data = {
        'num_lanes': num_lanes,
        'total_positions': total_positions,
        'lanes': trade_lanes
    }
    
    # New: Get relevant ports from commitments
    relevant_ports = get_relevant_ports_from_commitments(commitments, trade_code)
    
    # New: Load optimal TSP sequences per lane (from optimal TSP solver)
    # First try optimal sequences, fall back to filtered original if not available
    tsp_sequences = {}
    optimal_tsp_dir = base_dir / 'DATA/Master/Port_Sequences/3_Optimal_Filtered_TSP_Sequences'
    original_tsp_dir = base_dir / 'DATA/Master/Port_Sequences/1_Original_TSP_Sequence'
    
    for lane in trade_lanes:
        lane_name = lane['name']
        
        # Try optimal TSP first
        optimal_filename = f"optimal_subset_sequence_{lane_name}.csv"
        optimal_path = optimal_tsp_dir / optimal_filename
        
        if optimal_path.exists():
            # Load optimal TSP (already optimal, no filtering needed)
            optimal_tsp = load_optimal_tsp(optimal_path)
            if optimal_tsp is not None and optimal_tsp:
                tsp_sequences[lane_name] = optimal_tsp
            else:
                tsp_sequences[lane_name] = []
        else:
            # Fall back to original TSP with filtering
            original_filename = f"PortSequence_{lane_name}.csv"
            original_path = original_tsp_dir / original_filename
            
            filtered_tsp = load_and_filter_tsp(original_path, relevant_ports)
            if filtered_tsp is not None and filtered_tsp:
                tsp_sequences[lane_name] = filtered_tsp
            else:
                tsp_sequences[lane_name] = []  # Empty if no file or no matches
    
    # New: Build distance matrices for each lane (DAG arcs)
    distance_matrices = {}
    for lane_name, filtered_tsp in tsp_sequences.items():
        if filtered_tsp:
            distance_matrices[lane_name] = build_distance_matrix_for_lane(filtered_tsp, distances, ports)
        else:
            distance_matrices[lane_name] = {}
    
    # New: Save filtered TSPs + distance matrices to CSV for inspection
    import os
    filtered_dir = base_dir / 'DATA/Master/Port_Sequences/2_Filtered_TSP_Sequences'
    filtered_dir.mkdir(exist_ok=True)  # Create if needed

    for lane_name, filtered_tsp in tsp_sequences.items():
        if filtered_tsp:  # Only if non-empty
            # Save filtered TSP (ports only, sequence preserved)
            output_filename = f"NewPortSequence_{trade_code}_{lane_name}.csv"
            output_path = filtered_dir / output_filename
            
            # Create DataFrame - remove stale sailing_nm/sailing_days (replaced by distance matrix)
            filtered_df = pd.DataFrame(filtered_tsp)
            filtered_df = filtered_df[['port', 'sequence', 'sub_area']]
            filtered_df.to_csv(output_path, index=False)
            
            # Save distance matrix with timing data
            if lane_name in distance_matrices and distance_matrices[lane_name]:
                dist_matrix = distance_matrices[lane_name]
                matrix_filename = f"DistanceMatrix_{trade_code}_{lane_name}.csv"
                matrix_path = filtered_dir / matrix_filename
                # Save as edge list with all timing components
                matrix_df = pd.DataFrame([
                    {
                        'from_port': p_from,
                        'to_port': p_to,
                        'distance_nm': arc_data['distance_nm'],
                        'port_time_days': arc_data['port_time_days'],
                        'sailing_time_days': arc_data['sailing_time_days'],
                        'total_arc_time_days': arc_data['total_arc_time_days']
                    }
                    for (p_from, p_to), arc_data in sorted(dist_matrix.items())
                ])
                matrix_df.to_csv(matrix_path, index=False)
            

    return {
        'name': trade_data['name'],
        'min_spread': trade_data['min_spread'],
        'loading_ports': {
            'min': trade_data['min_loading_ports'],
            'max': trade_data['max_loading_ports']
        },
        'discharge_ports': {
            'min': trade_data['min_discharge_ports'],
            'max': trade_data['max_discharge_ports']
        },
        'route_start_range': route_start,
        'route_dead_position': route_dead,
        'transit_times': trade_data['transit_times'],
        'ports_with_constraints': ports_with_constraints,
        'commitments': trade_commitments,
        'requested_lanes': requested_lanes_data,
        'relevant_ports': list(relevant_ports),  # New: Explicit list for convenience
        'tsp_sequences': tsp_sequences,  # New: Filtered per-lane TSPs
        'distance_matrices': distance_matrices,  # New: Per-lane DAG arc costs
        'port_classifications': port_classifications # New: Port classifications
    }


def get_port_constraints_for_trade(ports, trade_code):
    """
    Returns ALL ports with their full data (constraints + base info) for a specific trade.
    This is the main function you'll use when building a MIP for a trade.
    
    Args:
        ports: ports dictionary from load_ports()
        trade_code: the specific trade code (e.g., 'EUME')
    
    Returns:
        Dictionary of {port_name: full_port_data} for ports with constraints in this trade.
        Each port includes both its constraints AND base information (area, call duration, etc.)
    
    Example:
        port_data = get_port_constraints_for_trade(ports, 'EUME')
        for port_name, data in port_data.items():
            print(f"{port_name}: spread={data['min_spread']}, area={data['area']}")
    """
    result = {}
    
    for port_name, port_data in ports.items():
        if trade_code in port_data['trade_constraints']:
            # Combine base port info with trade-specific constraints
            result[port_name] = {
                # Trade-specific constraints
                **port_data['trade_constraints'][trade_code],
                # Base port information
                'area': port_data['area'],
                'sub_area': port_data['sub_area'],
                'call_duration_days': port_data['call_duration_days'],
                'port_delays_days': port_data['port_delays_days'],
                'total_port_time_days': port_data['total_port_time_days'],
                'port_type': port_data['port_type'],
                'in_eu': port_data['in_eu'],
                'waypoint': port_data['waypoint'],
                'longitude': port_data['longitude'],
                'latitude': port_data['latitude']
            }
    
    return result


def get_transit_times_for_trade(trades, trade_code):
    """
    Returns all transit times for a given trade as a dictionary.
    Key: (port_from, port_to) tuple
    Value: maximum transit time in days
    
    Returns empty dict if trade not found or has no transit times.
    """
    if trade_code in trades:
        return trades[trade_code]['transit_times']
    return {}


# ================================
# MAIN EXECUTION (for testing)
# ================================
if __name__ == "__main__":
    # Load all global data unconditionally first
    trades = load_trades()
    ports = load_ports()
    commitments = load_commitments()
    distances = load_distances()
    requested_lanes = load_requested_lanes()

    if PRINT_TRADES:
        print("="*60)
        print("LOADING SERVICE CORRIDORS (TRADES)")
        print("="*60)
        print(f"Total number of trades: {len(trades)}")
        print(f"Trade codes: {list(trades.keys())}")
        
        print("\n" + "-"*60)
        print("Sample Trade Details:")
        print("-"*60)
        for trade_code in list(trades.keys())[:3]:
            print(f"\n{trade_code}: {trades[trade_code]['name']}")
            print(f"  Min/Max Loading Ports: {trades[trade_code]['min_loading_ports']}/{trades[trade_code]['max_loading_ports']}")
            print(f"  Min/Max Discharge Ports: {trades[trade_code]['min_discharge_ports']}/{trades[trade_code]['max_discharge_ports']}")
            print(f"  Min Spread: {trades[trade_code]['min_spread']}")
            print(f"  Route Start Range: {trades[trade_code]['route_start_range_lower']}-{trades[trade_code]['route_start_range_upper']}")

    if PRINT_PORTS:
        print("\n" + "="*60)
        print("LOADING PORTS")
        print("="*60)
        print(f"Total number of ports: {len(ports)}")
        
        print("\n" + "-"*60)
        print("Sample Port Details:")
        print("-"*60)
        for port_name in list(ports.keys())[:5]:
            print(f"\n{port_name}:")
            print(f"  Area: {ports[port_name]['area']}, Sub-area: {ports[port_name]['sub_area']}")
            print(f"  Port Type: {ports[port_name]['port_type']}")
            print(f"  Call Duration: {ports[port_name]['call_duration_days']} days")
            print(f"  Total Port Time: {ports[port_name]['total_port_time_days']} days")
            print(f"  Coordinates: ({ports[port_name]['latitude']}, {ports[port_name]['longitude']})")
            print(f"  In EU: {ports[port_name]['in_eu']}")
        
        print("\n" + "-"*60)
        print("Port Statistics:")
        print("-"*60)
        cargo_ops = sum(1 for p in ports.values() if p['port_type'] == 'Cargo Ops')
        canals = sum(1 for p in ports.values() if p['port_type'] == 'Canal')
        routing_points = sum(1 for p in ports.values() if p['port_type'] == 'Routing Point')
        eu_ports = sum(1 for p in ports.values() if p['in_eu'] == 'Y')
        print(f"Cargo Operations Ports: {cargo_ops}")
        print(f"Canals: {canals}")
        print(f"Routing Points: {routing_points}")
        print(f"EU Ports: {eu_ports}")
        
        areas = {}
        for port_name, port_data in ports.items():
            area = port_data['area']
            if area not in areas:
                areas[area] = []
            areas[area].append(port_name)
        
        print("\n" + "-"*60)
        print("Ports by Area:")
        print("-"*60)
        for area, port_list in sorted(areas.items()):
            print(f"{area}: {len(port_list)} ports")
        
        print("\n" + "-"*60)
        print("Ports with Trade-Specific Constraints:")
        print("-"*60)
        ports_with_constraints = [(name, data) for name, data in ports.items() if data['trade_constraints']]
        print(f"Total ports with trade constraints: {len(ports_with_constraints)}")
        
        print("\nDetailed constraint information:")
        for port_name, port_data in sorted(ports_with_constraints):
            print(f"\n{port_name}:")
            for trade, constraints in port_data['trade_constraints'].items():
                print(f"  {trade}:")
                print(f"    Min Spread: {constraints['min_spread']}")
                print(f"    Desired Position: {constraints['desired_position_lower']}-{constraints['desired_position_upper']}")
                print(f"    Port on USIN: {constraints['port_on_usin']}")

    if PRINT_COMMITMENTS:
        print("\n" + "="*60)
        print("LOADING COMMITMENTS, DISTANCES, AND LANES")
        print("="*60)
        print(f"Total commitments: {sum(len(c) for c in commitments.values())} port-pairs across {len(commitments)} trades")
        print(f"Total distance pairs: {len(distances)} (non-zero, symmetric)")

    if PRINT_LANES:
        print("\n" + "="*60)
        print("LOADING REQUESTED LANES")
        print("="*60)
        total_lanes_all = sum(len(lanes) for lanes in requested_lanes.values())
        total_positions_all = sum(sum(lane['positions'] for lane in lanes) for lanes in requested_lanes.values())
        print(f"Total lane variants: {total_lanes_all} across {len(requested_lanes)} trades")
        print(f"Total requested positions: {total_positions_all}")

    if PRINT_TSP:
        print("\n" + "="*60)
        print("TSP SEQUENCES OVERVIEW")
        print("="*60)
        tsp_dir = base_dir / 'DATA/Master/Port_Sequences/1_Original_TSP_Sequence'
        tsp_files = [f for f in tsp_dir.glob("PortSequence_*.csv")]
        print(f"Total TSP files available: {len(tsp_files)}")

    if PRINT_RELEVANT_PORTS:
        print("\n" + "="*60)
        print("RELEVANT PORTS FROM COMMITMENTS (PER TRADE)")
        print("="*60)
        for trade_code in get_trade_list(trades):
            rel_ports = get_relevant_ports_from_commitments(commitments, trade_code)
            num_rel = len(rel_ports)
            print(f"\n{trade_code} ({trades[trade_code]['name']}): {num_rel} unique ports")
            sorted_ports = sorted(rel_ports)
            for i in range(0, len(sorted_ports), 5):
                print(f"  {', '.join(sorted_ports[i:i+5])}")
            if num_rel == 0:
                print("  (No commitments—trivial TSP)")

    if PRINT_SUMMARY:
        print("\n" + "="*60)
        print("LOOPING THROUGH ALL TRADES (Summary with Commitments, Lanes & TSP)")
        print("="*60)
        for trade_code in get_trade_list(trades):
            trade_data = get_all_data_for_trade(trades, ports, commitments, requested_lanes, distances, trade_code)
            num_ports_const = len(trade_data['ports_with_constraints'])
            num_transit = len(trade_data['transit_times'])
            num_commit = len(trade_data['commitments'])
            num_lanes = trade_data['requested_lanes']['num_lanes']
            total_pos = trade_data['requested_lanes']['total_positions']
            num_relevant_ports = len(trade_data['relevant_ports'])
            num_tsp_lanes = len(trade_data['tsp_sequences'])
            num_arcs_total = sum(len(m) for m in trade_data['distance_matrices'].values())
            print(f"{trade_code}: {trade_data['name'][:40]:40} | "
                  f"Spread: {trade_data['min_spread']:2} | "
                  f"Ports(Const): {num_ports_const:2} | "
                  f"Rel Ports: {num_relevant_ports:2} | "
                  f"Transit: {num_transit:3} | "
                  f"Commit: {num_commit:2} | "
                  f"Lanes: {num_lanes:1} | "
                  f"Pos: {total_pos:1} | "
                  f"TSP Lanes: {num_tsp_lanes:1} | "
                  f"Arcs: {num_arcs_total:3}")
        
        print("\n" + "="*60)
        print("DATA STRUCTURES LOADED SUCCESSFULLY!")
        print("="*60)

    # Usage examples (always, or flag)
    print("\n" + "="*60)
    print("UPDATED USAGE EXAMPLES FOR MIP (WITH COMMITMENTS, DISTANCES, LANES & TSP)")
    print("="*60)
    print("\n# Load all global data once")
    print("from DataProcessing.Sets import load_trades, load_ports, load_commitments, load_distances, load_requested_lanes")
    print("from DataProcessing.Sets import get_trade_list, get_all_data_for_trade, NewTSP")
    print("")
    print("trades = load_trades()")
    print("ports = load_ports()")
    print("commitments = load_commitments()")
    print("distances = load_distances()")
    print("requested_lanes = load_requested_lanes()")
    print("")
    print("\n# Example 1: Loop trades")
    print("for trade_code in get_trade_list(trades):")
    print("    data = get_all_data_for_trade(trades, ports, commitments, requested_lanes, distances, trade_code)")
    print("    model = build_mip(trade_code, data, distances)")
    print("    model.optimize()")
    print("")
    print("\n# Example 2: Single trade with TSP for DAG")
    print("data = get_all_data_for_trade(trades, ports, commitments, requested_lanes, distances, 'EUAF')")
    print("")
    print("# Existing...")
    print("# New: Use filtered TSP for DAG graph")
    print("relevant_ports = set(data['relevant_ports'])")
    print("for lane_name, tsp_seq in data['tsp_sequences'].items():")
    print("    if tsp_seq:  # Has filtered ports")
    print("        # Build DAG edges: only increasing sequence")
    print("        for i in range(len(tsp_seq) - 1):")
    print("            port_i = tsp_seq[i]['port']")
    print("            port_j = tsp_seq[i+1]['port']")
    print("            seq_i = tsp_seq[i]['sequence']")
    print("            seq_j = tsp_seq[i+1]['sequence']")
    print("            if seq_j > seq_i:  # Enforce DAG")
    print("                # Add edge var x[(port_i, port_j, lane_name)]")
    print("                sail_days = tsp_seq[i]['sailing_days']")
    print("                model.addConstr(x[(port_i, port_j, lane_name)] * sail_days <= total_time)")
    print("                # Objective: min sum x * sailing_days + distances...")
    print("    # Standalone NewTSP usage")
    print("    tsp_path = 'DATA/Master/Port_Sequences/1_Original_TSP_Sequence/PortSequence_EUAFviaCapetoOC.csv'")
    print("    filtered = NewTSP(tsp_path, relevant_ports)")
    print("    print(f'Filtered: {len(filtered)} ports')")
    print("")
    print("# Fleet/lanes (unchanged)")
    print("total_pos = data['requested_lanes']['total_positions']")
    print("if total_pos > 0:")
    print("    model.addConstr(num_vessels >= total_pos)")
    print("")
    print("# Distances (unchanged)")
    print("speed_knots = 20")
    print("for (port_i, port_j) in possible_edges:")
    print("    if (port_i, port_j) in distances:")
    print("        dist_nm = distances[(port_i, port_j)]")
    print("        sail_time_days = dist_nm / (speed_knots * 24)")
    print("        model.addConstr(sail_time[(port_i, port_j)] == sail_time_days)")

