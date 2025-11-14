"""
Solve optimal TSP for relevant ports in a trade using Gurobi.
Finds the shortest route visiting all relevant ports exactly once.
"""

import sys
from pathlib import Path
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# Add parent directory to path to import Sets
sys.path.insert(0, str(Path(__file__).parent.parent))

from DataProcessing.Sets import (
    load_trades,
    load_ports,
    load_commitments,
    load_distances,
    load_requested_lanes,
    get_relevant_ports_from_commitments,
    normalize_port_name,
    load_and_filter_tsp
)


def solve_tsp_gurobi(ports_list, distances_dict):
    """
    Solve Open TSP (shortest path, not a loop) using Gurobi with dummy node trick.
    
    Uses the dummy node approach: adds a dummy node with 0 distances to/from all ports,
    solves a closed TSP on the expanded graph, then extracts the open path.
    
    Args:
        ports_list: List of port names to visit
        distances_dict: Dictionary {(port_from, port_to): distance_nm}
    
    Returns:
        List of port names in optimal visiting order (open path, not a loop), or None if infeasible
    """
    if len(ports_list) < 2:
        return ports_list  # Trivial case
    
    # Create expanded node list with dummy node
    DUMMY_NODE = "__DUMMY__"
    expanded_nodes = ports_list + [DUMMY_NODE]
    n = len(expanded_nodes)
    dummy_idx = n - 1  # Dummy is the last node
    
    # Create expanded distance matrix
    expanded_distances = {}
    for i, port_i in enumerate(expanded_nodes):
        for j, port_j in enumerate(expanded_nodes):
            if i != j:
                if i == dummy_idx or j == dummy_idx:
                    # Dummy node: 0 distance to/from all ports
                    expanded_distances[(i, j)] = 0.0
                else:
                    # Real ports: use actual distance
                    dist = None
                    if (port_i, port_j) in distances_dict:
                        dist = distances_dict[(port_i, port_j)]
                    elif (port_j, port_i) in distances_dict:
                        dist = distances_dict[(port_j, port_i)]
                    
                    if dist is not None:
                        expanded_distances[(i, j)] = dist
    
    # Create model
    model = gp.Model("OpenTSP")
    model.setParam('OutputFlag', 0)  # Suppress Gurobi output
    
    # Decision variables: x[i,j] = 1 if we go from node i to node j
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j and (i, j) in expanded_distances:
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    
    # MTZ variables: u[i] = position of node i in the tour (1 to n)
    u = {}
    for i in range(1, n):  # u[0] is fixed at 1
        u[i] = model.addVar(vtype=GRB.INTEGER, lb=2, ub=n, name=f"u_{i}")
    
    # Objective: minimize total distance
    obj = gp.quicksum(
        x[i, j] * expanded_distances[(i, j)]
        for i in range(n)
        for j in range(n)
        if i != j and (i, j) in x
    )
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Constraints: Each node must be entered exactly once
    for j in range(n):
        model.addConstr(
            gp.quicksum(x[i, j] for i in range(n) if i != j and (i, j) in x) == 1,
            name=f"enter_{j}"
        )
    
    # Constraints: Each node must be left exactly once
    for i in range(n):
        model.addConstr(
            gp.quicksum(x[i, j] for j in range(n) if i != j and (i, j) in x) == 1,
            name=f"leave_{i}"
        )
    
    # MTZ subtour elimination constraints: u[i] - u[j] + n*x[i,j] <= n-1
    for i in range(1, n):
        for j in range(1, n):
            if i != j and (i, j) in x:
                model.addConstr(
                    u[i] - u[j] + n * x[i, j] <= n - 1,
                    name=f"mtz_{i}_{j}"
                )
    
    # Optimize
    model.optimize()
    
    if model.Status != GRB.OPTIMAL:
        print(f"TSP solve failed with status: {model.Status}")
        return None
    
    # Extract tour (closed loop including dummy)
    tour = [0]  # Start at node 0
    current = 0
    
    while len(tour) < n:
        for j in range(n):
            if j != current and (current, j) in x and x[current, j].X > 0.5:
                tour.append(j)
                current = j
                break
    
    # Find dummy node position in tour
    dummy_pos = tour.index(dummy_idx)
    
    # Extract open path: the segment that doesn't include dummy
    # The path goes from the port after dummy to the port before dummy (wrapping around)
    if dummy_pos == 0:
        # Dummy is first, path is from second to last
        open_path_indices = tour[1:]
    elif dummy_pos == len(tour) - 1:
        # Dummy is last, path is from first to second-to-last
        open_path_indices = tour[:-1]
    else:
        # Dummy is in the middle, path wraps around
        # Path: nodes after dummy -> nodes before dummy
        open_path_indices = tour[dummy_pos + 1:] + tour[:dummy_pos]
    
    # Convert indices to port names (exclude dummy)
    optimal_sequence = [expanded_nodes[i] for i in open_path_indices if i != dummy_idx]
    
    return optimal_sequence


def calculate_sailing_time(distance_nm):
    """
    Calculate sailing time in days based on distance.
    
    Args:
        distance_nm: Distance in nautical miles
    
    Returns:
        Sailing time in days
    """
    if distance_nm < 400:
        speed_knots = 14.0  # Coastal/short sailing
    else:
        speed_knots = 15.5  # Open ocean
    
    # Sailing time = distance / (speed * 24 hours)
    time_days = distance_nm / (speed_knots * 24)
    return time_days


def solve_tsp_for_lane(lane_name, trade_code=None, output_filename=None):
    """
    Solve TSP for relevant ports in a lane and save to CSV.
    
    The ports come from the lane's TSP sequence file, filtered to only include
    ports that are in the trade's commitments (relevant ports).
    Distances come from the global distance matrix.
    
    Args:
        lane_name: Lane name (e.g., 'EUAFviaCapetoOC', 'EUMEviaCape')
        trade_code: Optional trade code (auto-detected if not provided)
        output_filename: Optional output filename (default: optimal_subset_sequence_{lane_name}.csv)
    
    Returns:
        Path to output file, or None if failed
    """
    print(f"Solving TSP for lane: {lane_name}")
    
    # Load all data
    print("Loading data...")
    base_dir = Path(__file__).parent.parent
    trades = load_trades()
    ports = load_ports()
    commitments = load_commitments()
    distances = load_distances()
    requested_lanes = load_requested_lanes()
    
    # Auto-detect trade code if not provided
    if trade_code is None:
        for trade, lanes in requested_lanes.items():
            if any(lane['name'] == lane_name for lane in lanes):
                trade_code = trade
                break
        if trade_code is None:
            print(f"ERROR: Could not find trade code for lane {lane_name}")
            return None
    
    print(f"Trade code: {trade_code}")
    
    # Get relevant ports for this trade (from commitments)
    relevant_ports_set = get_relevant_ports_from_commitments(commitments, trade_code)
    
    if not relevant_ports_set:
        print(f"Warning: No relevant ports found for trade {trade_code}")
        return None
    
    # Load the lane's TSP sequence file to get all ports in this lane
    tsp_dir = base_dir / 'DATA/TSP_sequence_per_lane'
    tsp_filename = f"PortSequence_{lane_name}.csv"
    tsp_path = tsp_dir / tsp_filename
    
    if not tsp_path.exists():
        print(f"ERROR: TSP sequence file not found: {tsp_path}")
        return None
    
    # Load and filter TSP sequence to only relevant ports
    filtered_tsp = load_and_filter_tsp(tsp_path, relevant_ports_set)
    
    if not filtered_tsp:
        print(f"Warning: No relevant ports found in lane {lane_name} TSP sequence")
        return None
    
    # Extract port list from filtered TSP (these are the ports we'll solve TSP for)
    lane_ports_list = [entry['port'] for entry in filtered_tsp]
    print(f"Found {len(lane_ports_list)} relevant ports in lane: {lane_ports_list}")
    
    # Check that we have distances for all port pairs (required for TSP)
    missing_distances = []
    for i, port_i in enumerate(lane_ports_list):
        for j, port_j in enumerate(lane_ports_list):
            if i != j:
                if (port_i, port_j) not in distances and (port_j, port_i) not in distances:
                    missing_distances.append((port_i, port_j))
    
    if missing_distances:
        print(f"ERROR: Missing distances for {len(missing_distances)} port pairs (TSP requires all pairs):")
        for p1, p2 in missing_distances[:10]:  # Show first 10
            print(f"  {p1} -> {p2}")
        if len(missing_distances) > 10:
            print(f"  ... and {len(missing_distances) - 10} more")
        print("Cannot solve TSP without complete distance matrix.")
        return None
    
    # Solve TSP
    print("Solving TSP...")
    optimal_sequence = solve_tsp_gurobi(lane_ports_list, distances)
    
    if optimal_sequence is None:
        print("Failed to solve TSP")
        return None
    
    print(f"Optimal sequence found: {' -> '.join(optimal_sequence)}")
    
    # Build output data
    output_data = []
    for idx, port in enumerate(optimal_sequence):
        seq_num = idx + 1  # 1-based sequence number for output
        # Get sub-area from ports data
        sub_area = ports.get(port, {}).get('sub_area', '')
        
        # Calculate distance and time to next port
        if idx < len(optimal_sequence) - 1:
            next_port = optimal_sequence[idx + 1]
            # Try both directions for distance
            distance_nm = distances.get((port, next_port))
            if distance_nm is None:
                distance_nm = distances.get((next_port, port), 0.0)
        else:
            # Last port: distance and time are 0
            distance_nm = 0.0
        
        time_days = calculate_sailing_time(distance_nm) if distance_nm > 0 else 0.0
        
        output_data.append({
            'Port': port,
            'Sequence': seq_num,
            'Sub-area': sub_area,
            'distance_next_port': distance_nm,
            'time_to_next_port': time_days
        })
    
    # Create DataFrame and save
    df_output = pd.DataFrame(output_data)
    
    # Determine output path (save to Optimal_Filtered_TSP_Sequences folder)
    if output_filename is None:
        output_filename = f"optimal_subset_sequence_{lane_name}.csv"
    
    output_dir = base_dir / 'DATA' / 'Optimal_Filtered_TSP_Sequences'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename
    
    df_output.to_csv(output_path, index=False)
    print(f"Output saved to: {output_path}")
    
    return output_path


def solve_tsp_for_all_lanes():
    """
    Solve TSP for all lanes automatically.
    Processes all lanes from Requested_Lanes.csv and saves results to Optimal_Filtered_TSP_Sequences.
    """
    print("="*70)
    print("SOLVING TSP FOR ALL LANES")
    print("="*70)
    
    # Load data once
    base_dir = Path(__file__).parent.parent
    requested_lanes = load_requested_lanes()
    
    # Collect all lanes
    all_lanes = []
    for trade_code, lanes in requested_lanes.items():
        for lane in lanes:
            all_lanes.append((lane['name'], trade_code))
    
    print(f"\nFound {len(all_lanes)} lanes to process:\n")
    for lane_name, trade_code in all_lanes:
        print(f"  {lane_name} ({trade_code})")
    
    print(f"\n{'='*70}\n")
    
    # Process each lane
    results = {}
    for idx, (lane_name, trade_code) in enumerate(all_lanes, 1):
        print(f"\n[{idx}/{len(all_lanes)}] Processing: {lane_name}")
        print("-" * 70)
        
        try:
            result_path = solve_tsp_for_lane(lane_name, trade_code, output_filename=None)
            if result_path:
                results[lane_name] = {'status': 'SUCCESS', 'path': result_path}
            else:
                results[lane_name] = {'status': 'FAILED', 'path': None}
        except Exception as e:
            print(f"ERROR processing {lane_name}: {e}")
            results[lane_name] = {'status': 'ERROR', 'error': str(e), 'path': None}
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    success_count = sum(1 for r in results.values() if r['status'] == 'SUCCESS')
    failed_count = len(results) - success_count
    
    print(f"Total lanes processed: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    
    if failed_count > 0:
        print(f"\nFailed lanes:")
        for lane_name, result in results.items():
            if result['status'] != 'SUCCESS':
                error_msg = result.get('error', 'Unknown error')
                print(f"  {lane_name}: {error_msg}")
    
    print(f"\nResults saved to: {base_dir / 'DATA' / 'Optimal_Filtered_TSP_Sequences'}")
    print(f"{'='*70}\n")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Solve optimal TSP for relevant ports in lanes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all lanes automatically
  python NEWTSP.py --all
  
  # Process a specific lane
  python NEWTSP.py EUAFviaCapetoOC
        """
    )
    parser.add_argument('lane_name', type=str, nargs='?', default=None,
                       help='Lane name (e.g., EUAFviaCapetoOC). If not provided, use --all to process all lanes.')
    parser.add_argument('--all', action='store_true',
                       help='Process all lanes automatically')
    parser.add_argument('--trade', '-t', type=str, default=None,
                       help='Trade code (auto-detected if not provided)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output filename (default: optimal_subset_sequence_{lane_name}.csv)')
    
    args = parser.parse_args()
    
    if args.all or args.lane_name is None:
        # Process all lanes
        solve_tsp_for_all_lanes()
    else:
        # Process single lane
        solve_tsp_for_lane(args.lane_name, args.trade, args.output)

