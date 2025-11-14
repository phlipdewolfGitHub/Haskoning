import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from DataProcessing.Sets import (
    load_trades,
    load_ports,
    load_commitments,
    load_distances,
    load_requested_lanes,
    get_all_data_for_trade,
)

import gurobipy as gp
from gurobipy import GRB

# Configuration constants
CYCLE_DAYS = 30
MAX_ROUTE_DURATION = 200  # Conservative bound on route duration (days)
MAX_WRAP_COUNT = math.ceil(MAX_ROUTE_DURATION / CYCLE_DAYS)

# Constraint toggles (can be overridden)

ENFORCE_START_SPREAD = True  # C2: Minimum trade start spread between positions
ENFORCE_PORT_SPREAD = True  # C3: Minimum port spread within trade
ENFORCE_START_RANGE = True  # C4: Preferred start range (at least one position)
ENFORCE_DEATH_AVOIDANCE = True  # C5: Death position avoidance
ENFORCE_PORT_WINDOWS = True  # C7: Target visit range for ports (at least one position per port)
ENFORCE_LOADING_DISCHARGING = True  # C8,9: Min/max loading/discharging constraints
ENFORCE_MAX_TRANSIT = True  # C10: Maximum transit time for served commitments


def build_and_solve_mip(trade_code, verbose_level=0,
                        enforce_loading_discharging=None,
                        enforce_start_spread=None,
                        enforce_port_spread=None,
                        enforce_start_range=None,
                        enforce_death_avoidance=None,
                        enforce_port_windows=None,
                        enforce_max_transit=None):
    """
    Build and solve MIP model for a given trade.
    
    Args:
        trade_code: Trade code (e.g., 'FEUS', 'EUAF')
        verbose_level: 0: Minimal output, 1: Basic output, 2: Detailed output
        enforce_*: Optional overrides for constraint toggles
    
    Returns:
        Dictionary with keys:
            - 'status': Model status string ('OPTIMAL', 'INFEASIBLE', etc.)
            - 'objective': Objective value (if solution found)
            - 'routes': List of route dictionaries, each with:
                - 'position_id': Position ID tuple
                - 'lane_name': Lane name
                - 'start_day': Route start day
                - 'visits': List of visit dicts with 'port', 'arrival_day', 'day_of_month'
    """
    # Use provided overrides or defaults
    enforce_c2 = enforce_start_spread if enforce_start_spread is not None else ENFORCE_START_SPREAD
    enforce_c3 = enforce_port_spread if enforce_port_spread is not None else ENFORCE_PORT_SPREAD
    enforce_c4 = enforce_start_range if enforce_start_range is not None else ENFORCE_START_RANGE
    enforce_c5 = enforce_death_avoidance if enforce_death_avoidance is not None else ENFORCE_DEATH_AVOIDANCE
    enforce_c7 = enforce_port_windows if enforce_port_windows is not None else ENFORCE_PORT_WINDOWS
    enforce_c89 = enforce_loading_discharging if enforce_loading_discharging is not None else ENFORCE_LOADING_DISCHARGING
    enforce_c10 = enforce_max_transit if enforce_max_transit is not None else ENFORCE_MAX_TRANSIT
    model = gp.Model(f"MIP_{trade_code}")
    model.Params.OutputFlag = 1 if verbose_level > 1 else 0
    if verbose_level > 0:
        print("\n" + "="*60)
        print(" " * 10 + "***   MIP ROUTE OPTIMIZATION   ***")
        print("="*60)
        
    
    if verbose_level > 0:
        print("Loading data...\n")   
        print(f"Creating model for {trade_code}...\n") 
    trades = load_trades()
    ports = load_ports()
    commitments = load_commitments()
    distances = load_distances()
    requested_lanes = load_requested_lanes()

    data = get_all_data_for_trade(
        trades,
        ports,
        commitments,
        requested_lanes,
        distances,
        trade_code
    )

    # Build per-lane DAGs for the service corridor
    # Each lane type has its own set of ports/arcs, but shares the service corridor
    lanes_data = data["requested_lanes"]["lanes"]
    SOURCE_NODE = "__source__"
    SINK_NODE = "__sink__"

    # Build lane-specific node/arc sets
    lane_graphs = {}
    for lane_idx, lane_info in enumerate(lanes_data):
        lane_name = lane_info["name"]
        tsp_sequence = data["tsp_sequences"].get(lane_name, [])
        
        if not tsp_sequence:
            if verbose:
                print(f"Warning: Lane {lane_name} has no TSP sequence, skipping")
            continue
        
        ports = [entry["port"] for entry in tsp_sequence]
        nodes = [SOURCE_NODE] + ports + [SINK_NODE]
        
        # Build arcs for this lane and extract timing data
        dist_matrix = data["distance_matrices"].get(lane_name, {})
        port_arcs = list(dist_matrix.keys())
        source_arcs = [(SOURCE_NODE, p) for p in ports]
        sink_arcs = [(p, SINK_NODE) for p in ports]
        arcs = source_arcs + port_arcs + sink_arcs
        
        # Extract tau (total arc time) for each arc from distance matrix
        arc_times = {}
        for (i, j), arc_data in dist_matrix.items():
            arc_times[(i, j)] = arc_data['total_arc_time_days']
        
        # For source arcs to first port set time to 0
        port_data = data["ports_with_constraints"]
        for (src, p) in source_arcs:
            arc_times[(src, p)] = 0.0  
        
        # For sink arcs, no additional time needed (route ends)
        for (p, snk) in sink_arcs:
            arc_times[(p, snk)] = 0.0
        
        lane_graphs[lane_idx] = {
            "name": lane_name,
            "nodes": nodes,
            "arcs": arcs,
            "ports": ports,
            "num_positions": lane_info["positions"],
            "arc_times": arc_times,  # tau values for timing constraints
        }

    # Create position index: each lane has multiple positions (separate schedules)
    # Position IDs are tuples (lane_idx, pos_within_lane)
    positions = []
    for lane_idx, lane_info in enumerate(lanes_data):
        if lane_idx not in lane_graphs:
            continue  # Skip lanes with no TSP
        
        num_pos = lane_info["positions"]
        for pos_idx in range(num_pos):
            positions.append({
                "id": (lane_idx, pos_idx),
                "lane_idx": lane_idx,
                "lane_name": lane_info["name"],
                "pos_in_lane": pos_idx,
            })

    num_positions = len(positions)

    # Commitments for this trade (positive frequency only)
    trade_commitments = data.get("commitments", {})
    commitments_list = [
        {
            "id": idx,
            "from": c_from,
            "to": c_to,
            "frequency": freq,
        }
        for idx, ((c_from, c_to), freq) in enumerate(sorted(trade_commitments.items()))
        if freq > 0
    ]

    if num_positions == 0:
        raise ValueError(f"{trade_code} has no positions to schedule")
    
        
    # Decision variables per position
    # e[l, (i,j)]: arc selection for position l on its lane-specific arc set
    if verbose_level > 0:
        print(f"Setting up decision variables...\n")
    pos_ids = [p["id"] for p in positions]
    e = {}
    for pos_info in positions:
        pos_id = pos_info["id"]
        lane_idx = pos_info["lane_idx"]
        arcs = lane_graphs[lane_idx]["arcs"]
        ports = lane_graphs[lane_idx]["ports"]
        seq_arcs = []
        prev = SOURCE_NODE
        for port in ports:
            seq_arcs.append((prev, port))
            prev = port
        seq_arcs.append((ports[-1], SINK_NODE))

        for arc in arcs:
            var = e.setdefault((pos_id, arc[0], arc[1]), model.addVar(vtype=GRB.BINARY, name=f"e[{pos_id},{arc}]") )
            if arc in seq_arcs:
                var.Start = 1
            else:
                var.Start = 0

    # s[l]: start day per position
    s = model.addVars(pos_ids, vtype=GRB.INTEGER, lb=1, ub=CYCLE_DAYS, name="s")
    for pos_id in pos_ids:
        s[pos_id].Start = 1

    # Derived variables: only for ports in each position's lane
    x = {}
    alpha = {}
    phi = {}
    w = {}

    for pos_info in positions:
        pos_id = pos_info["id"]
        lane_idx = pos_info["lane_idx"]
        ports = lane_graphs[lane_idx]["ports"]
        
        for port in ports:
            x[pos_id, port] = model.addVar(vtype=GRB.BINARY, name=f"x[{pos_id},{port}]")
            alpha[pos_id, port] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"alpha[{pos_id},{port}]")
            phi[pos_id, port] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=CYCLE_DAYS, name=f"phi[{pos_id},{port}]")
            w[pos_id, port] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=MAX_WRAP_COUNT, name=f"w[{pos_id},{port}]")

    # Commitment coverage indicators z[pos, commitment]
    z = {}
    for pos_info in positions:
        pos_id = pos_info["id"]
        lane_idx = pos_info["lane_idx"]
        ports_set = set(lane_graphs[lane_idx]["ports"])
        for commit in commitments_list:
            if commit["from"] in ports_set and commit["to"] in ports_set:
                cid = commit["id"]
                z[pos_id, cid] = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"z[{pos_id},{commit['from']}->{commit['to']}]"
                )

    # New constraint variables (C2, C3, C4, C5)
    # C2: Binary ordering variables for start spread
    b = {}
    if ENFORCE_START_SPREAD and num_positions > 1:
        b = model.addVars(
            [(pos_ids[i], pos_ids[j]) for i in range(len(pos_ids)) for j in range(i+1, len(pos_ids))],
            vtype=GRB.BINARY,
            name="b"
        )

    # C3: Binary ordering variables for port spread
    z_spread = {}

    # C4: Binary indicators for which position is in start range
    y_start_range = {}
    if ENFORCE_START_RANGE and data.get('route_start_range') is not None:
        y_start_range = model.addVars(pos_ids, vtype=GRB.BINARY, name="y_start_range")

    # C5: Binary indicators for death position avoidance
    h = {}
    if ENFORCE_DEATH_AVOIDANCE and data.get('route_dead_position') is not None:
        h = model.addVars(pos_ids, vtype=GRB.BINARY, name="h")

    # C7: Binary indicators for which position satisfies port window
    y_port_window = {}

    

    model.update()
    
    if verbose_level > 0:
        print("Adding constraints...\n")
    # Constraints
    # (1) Flow balance: single path from source to sink per position
    for pos_info in positions:
        pos_id = pos_info["id"]
        lane_idx = pos_info["lane_idx"]
        ports = lane_graphs[lane_idx]["ports"]
        
        # Exactly one arc leaving source
        source_out = [e[pos_id, SOURCE_NODE, p] for p in ports if (pos_id, SOURCE_NODE, p) in e]
        model.addConstr(gp.quicksum(source_out) == 1, name=f"source_out[{pos_id}]")
        
        # Exactly one arc entering sink
        sink_in = [e[pos_id, p, SINK_NODE] for p in ports if (pos_id, p, SINK_NODE) in e]
        model.addConstr(gp.quicksum(sink_in) == 1, name=f"sink_in[{pos_id}]")
        
        # Flow conservation at each port: inflow = outflow
        lane_arcs = lane_graphs[lane_idx]["arcs"]
        port_order = {p: idx for idx, p in enumerate(ports)}
        for port in ports:
            incoming = [e[pos_id, i, port] for i, j in lane_arcs if j == port and (pos_id, i, port) in e]
            outgoing = [e[pos_id, port, j] for i, j in lane_arcs if i == port and (pos_id, port, j) in e]

            if incoming and outgoing:
                model.addConstr(gp.quicksum(incoming) == gp.quicksum(outgoing),
                                name=f"flow[{pos_id},{port}]")

        # Skip constraints: if arc (i,j) jumps over intermediate ports, those ports cannot be visited
        for (i_node, j_node) in lane_arcs:
            if (pos_id, i_node, j_node) not in e:
                continue
            if i_node in {SOURCE_NODE, SINK_NODE} or j_node in {SOURCE_NODE, SINK_NODE}:
                continue
            if i_node not in port_order or j_node not in port_order:
                continue
            i_idx = port_order[i_node]
            j_idx = port_order[j_node]
            if j_idx <= i_idx + 1:
                continue  # adjacent ports, nothing to skip
            for skipped_port in ports[i_idx + 1:j_idx]:
                model.addConstr(
                    x[pos_id, skipped_port] <= 1 - e[pos_id, i_node, j_node],
                    name=f"skip[{pos_id},{i_node},{j_node}->{skipped_port}]"
                )

    # (2) Visit indicator: x[pos,p] = sum of incoming arcs (flow conservation ensures this equals outgoing)
    for pos_info in positions:
        pos_id = pos_info["id"]
        lane_idx = pos_info["lane_idx"]
        ports = lane_graphs[lane_idx]["ports"]
        
        for port in ports:
            # Use incoming arcs only (flow conservation already ensures incoming = outgoing)
            incoming = [e[pos_id, i, port] for i, j in lane_graphs[lane_idx]["arcs"] 
                    if j == port and (pos_id, i, port) in e]
            
            if incoming:
                model.addConstr(x[pos_id, port] == gp.quicksum(incoming), 
                            name=f"visit[{pos_id},{port}]")
            model.addConstr(
                alpha[pos_id, port] <= MAX_ROUTE_DURATION * x[pos_id, port],
                name=f"alpha_cap[{pos_id},{port}]"
            )

    # (3) Timing recursion: alpha[pos,q] = alpha[pos,p] + tau[(p,q)] if arc (p,q) selected
    # tau values now loaded from preprocessed data (sailing + port time)
    # Using BOTH inequalities to enforce exact travel times (no slack allowed)
    M_time = MAX_ROUTE_DURATION  # Big-M for timing (conservative upper bound on route duration in days)

    for pos_info in positions:
        pos_id = pos_info["id"]
        lane_idx = pos_info["lane_idx"]
        arc_times = lane_graphs[lane_idx]["arc_times"]
        ports = lane_graphs[lane_idx]["ports"]

        for (i, j) in lane_graphs[lane_idx]["arcs"]:
            if (pos_id, i, j) not in e:
                continue

            tau_arc = arc_times.get((i, j), 0.0)
            if i == SOURCE_NODE:
                if (pos_id, j) in alpha:
                    # First port: alpha must be exactly 0 when arc is selected
                    # The route starts on day s, and we immediately go to the first port
                    # Port handling time is accounted for in the port visit, not in arrival time
                    # Lower bound: alpha >= 0 when e=1, or >= -M when e=0
                    model.addConstr(alpha[pos_id, j] >= 0 - M_time * (1 - e[pos_id, i, j]),
                                name=f"alpha_init_lb[{pos_id},{j}]")
                    # Upper bound: alpha <= 0 when e=1, or <= M when e=0
                    model.addConstr(alpha[pos_id, j] <= 0 + M_time * (1 - e[pos_id, i, j]),
                                name=f"alpha_init_ub[{pos_id},{j}]")
            elif j == SINK_NODE:
                continue
            else:
                # Subsequent ports: alpha[j] = alpha[i] + tau (if arc selected)
                # Lower bound: alpha[j] >= alpha[i] + tau (when e=1)
                model.addConstr(
                    alpha[pos_id, j] >= alpha[pos_id, i] + tau_arc - M_time * (1 - e[pos_id, i, j]),
                    name=f"timing_lb[{pos_id},{i},{j}]"
                )
                # Upper bound: alpha[j] <= alpha[i] + tau (when e=1)
                model.addConstr(
                    alpha[pos_id, j] <= alpha[pos_id, i] + tau_arc + M_time * (1 - e[pos_id, i, j]),
                    name=f"timing_ub[{pos_id},{i},{j}]"
                )

    # (5) Modulo constraint: phi[pos,p] + CYCLE_DAYS*w[pos,p] = s[pos] + alpha[pos,p]

    for pos_info in positions:
        pos_id = pos_info["id"]
        lane_idx = pos_info["lane_idx"]
        ports = lane_graphs[lane_idx]["ports"]

        for port in ports:
            if (pos_id, port) in phi:
                model.addConstr(
                    phi[pos_id, port] + CYCLE_DAYS * w[pos_id, port] == (s[pos_id]-1) + alpha[pos_id, port],
                    name=f"modulo[{pos_id},{port}]"
                )

    


    # C1 Commitment feasibility: z <= visit indicators for both ports
    for commit in commitments_list:
        cid = commit["id"]
        from_port = commit["from"]
        to_port = commit["to"]
        covering = []
        for pos_info in positions:
            pos_id = pos_info["id"]
            if (pos_id, cid) not in z:
                continue
            covering.append(z[pos_id, cid])
            model.addConstr(
                z[pos_id, cid] <= x[pos_id, from_port],
                name=f"commit_from[{pos_id},{from_port}->{to_port}]"
            )
            model.addConstr(
                z[pos_id, cid] <= x[pos_id, to_port],
                name=f"commit_to[{pos_id},{from_port}->{to_port}]"
            )
        if covering:
            model.addConstr(
                gp.quicksum(covering) >= commit["frequency"],
                name=f"coverage[{from_port}->{to_port}]"
            )
    # (C2) Minimum trade start spread
    if ENFORCE_START_SPREAD and num_positions > 1:
        min_spread = data['min_spread']
        count_c2 = 0
        for i in range(len(pos_ids)):
            for j in range(i+1, len(pos_ids)):
                pos_id_1 = pos_ids[i]
                pos_id_2 = pos_ids[j]
                model.addConstr(
                    s[pos_id_1] - s[pos_id_2] + CYCLE_DAYS * b[pos_id_1, pos_id_2] >= min_spread,
                    name=f"start_spread_1[{pos_id_1},{pos_id_2}]"
                )
                model.addConstr(
                    s[pos_id_2] - s[pos_id_1] + CYCLE_DAYS * (1 - b[pos_id_1, pos_id_2]) >= min_spread,
                    name=f"start_spread_2[{pos_id_1},{pos_id_2}]"
                )
                count_c2 += 2
        if verbose_level > 0:
            print(f"Added {count_c2} C2 start spread constraints (min_spread={min_spread})")

    # C3 Minimum port spread within trade
    if ENFORCE_PORT_SPREAD:
        ports_with_constraints = data['ports_with_constraints']
        count_c3 = 0
        for port in ports_with_constraints:
            port_min_spread = ports_with_constraints[port]['min_spread']
            if port_min_spread > 0:
                # Find all positions that can visit this port
                pos_ids_for_port = [p for p in pos_ids if (p, port) in phi]
                # Add constraints for each ordered pair
                for i in range(len(pos_ids_for_port)):
                    for j in range(i+1, len(pos_ids_for_port)):
                        pos_id_1 = pos_ids_for_port[i]
                        pos_id_2 = pos_ids_for_port[j]
                        # Create z_spread variable for this pair
                        z_spread[pos_id_1, pos_id_2, port] = model.addVar(
                            vtype=GRB.BINARY,
                            name=f"z_spread[{pos_id_1},{pos_id_2},{port}]"
                        )
                        model.update()
                        # Add constraints
                        model.addConstr(
                            phi[pos_id_1, port] - phi[pos_id_2, port] + CYCLE_DAYS * z_spread[pos_id_1, pos_id_2, port]
                            >= port_min_spread - CYCLE_DAYS * (2 - x[pos_id_1, port] - x[pos_id_2, port]),
                            name=f"port_spread_1[{pos_id_1},{pos_id_2},{port}]"
                        )
                        model.addConstr(
                            phi[pos_id_2, port] - phi[pos_id_1, port] + CYCLE_DAYS * (1 - z_spread[pos_id_1, pos_id_2, port])
                            >= port_min_spread - CYCLE_DAYS * (2 - x[pos_id_1, port] - x[pos_id_2, port]),
                            name=f"port_spread_2[{pos_id_1},{pos_id_2},{port}]"
                        )
                        count_c3 += 2
        if verbose_level > 0:
            print(f"Added {count_c3} C3 port spread constraints")

    # C4 Preferred start range (at least one position)
    if ENFORCE_START_RANGE and data.get('route_start_range') is not None:
        route_start_range = data['route_start_range']
        lower = route_start_range['lower']
        upper = route_start_range['upper']
        count_c4 = 0
        # At least one position must be designated
        model.addConstr(gp.quicksum(y_start_range[pos_id] for pos_id in pos_ids) >= 1, 
                    name="start_range_at_least_one")
        count_c4 += 1
        # If a position is designated (y=1), its start must be in range
        for pos_id in pos_ids:
            model.addConstr(
                s[pos_id] >= lower - CYCLE_DAYS * (1 - y_start_range[pos_id]),
                name=f"start_range_lower[{pos_id}]"
            )
            model.addConstr(
                s[pos_id] <= upper + CYCLE_DAYS * (1 - y_start_range[pos_id]),
                name=f"start_range_upper[{pos_id}]"
            )
            count_c4 += 2
        if verbose_level > 0:
            print(f"Added {count_c4} C4 start range constraints (range=[{lower},{upper}], at least 1 position)")

    # C5 Death position avoidance
    if ENFORCE_DEATH_AVOIDANCE and data.get('route_dead_position') is not None:
        route_dead = data['route_dead_position']
        dead_lower = route_dead['lower']
        dead_upper = route_dead['upper']
        count_c5 = 0
        for pos_id in pos_ids:
            model.addConstr(
                s[pos_id] <= dead_lower - 1 + CYCLE_DAYS * h[pos_id],
                name=f"death_below[{pos_id}]"
            )
            model.addConstr(
                s[pos_id] >= dead_upper + 1 - CYCLE_DAYS * (1 - h[pos_id]),
                name=f"death_above[{pos_id}]"
            )
            count_c5 += 2
        if verbose_level > 0:
            print(f"Added {count_c5} C5 death avoidance constraints (range=[{dead_lower},{dead_upper}])")

    # C7 Target visit range for ports (at least one position per port)
    if ENFORCE_PORT_WINDOWS:
        ports_with_constraints = data['ports_with_constraints']
        count_c7 = 0
        for port in ports_with_constraints:
            desired_lower = ports_with_constraints[port]['desired_position_lower']
            desired_upper = ports_with_constraints[port]['desired_position_upper']
            if desired_lower > 0 or desired_upper > 0:
                # Find all positions that can visit this port
                pos_ids_for_port = [p for p in pos_ids if (p, port) in phi]
                
                if pos_ids_for_port:
                    # At least one position must satisfy the window
                    for pos_id in pos_ids_for_port:
                        y_port_window[pos_id, port] = model.addVar(
                            vtype=GRB.BINARY, 
                            name=f"y_window[{pos_id},{port}]"
                        )
                    model.update()
                    
                    # At least one designated position for this port
                    model.addConstr(
                        gp.quicksum(y_port_window[pos_id, port] for pos_id in pos_ids_for_port) >= 1,
                        name=f"port_window_at_least_one[{port}]"
                    )
                    count_c7 += 1
                    
                    # If a position is designated (y=1), it must visit the port within the window
                    for pos_id in pos_ids_for_port:
                        # y=1 implies x=1 (must visit)
                        model.addConstr(
                            y_port_window[pos_id, port] <= x[pos_id, port],
                            name=f"port_window_visit[{pos_id},{port}]"
                        )
                        # y=1 implies phi in window
                        model.addConstr(
                            phi[pos_id, port] >= (desired_lower-1) - CYCLE_DAYS * (1 - y_port_window[pos_id, port]),
                            name=f"port_window_lower[{pos_id},{port}]"
                        )
                        model.addConstr(
                            phi[pos_id, port] <= (desired_upper-1) + CYCLE_DAYS * (1 - y_port_window[pos_id, port]),
                            name=f"port_window_upper[{pos_id},{port}]"
                        )
                        count_c7 += 3
        if verbose_level > 0:
            print(f"Added {count_c7} C7 port window constraints (at least 1 position per port)")

    
    # (C8, C9 Min/max loading and discharging ports per position (capacity proxy)
    if ENFORCE_LOADING_DISCHARGING:
        if verbose_level > 0:
            print(f"Added {sum(1 for p in data['port_classifications']['loading_ports']) } C8 loading ports constraints")
            print(f"Added {sum(1 for p in data['port_classifications']['discharging_ports']) } C9 discharging ports constraints")
        for pos_info in positions:
            pos_id = pos_info["id"]
            lane_idx = pos_info["lane_idx"]
            lane_ports = set(lane_graphs[lane_idx]["ports"])
            
            # Trade-level min/max from data
            min_loading = data['loading_ports']['min']
            max_loading = data['loading_ports']['max']
            min_discharging = data['discharge_ports']['min']
            max_discharging = data['discharge_ports']['max']
            
            # Loading ports: intersection of trade loading ports and this lane's ports
            trade_loading_ports = set(data['port_classifications']['loading_ports'])
            loading_ports_in_lane = list(trade_loading_ports & lane_ports)
            loading_sum = gp.quicksum(x[pos_id, p] for p in loading_ports_in_lane)
            
            if loading_ports_in_lane:  # Only add if relevant ports exist
                model.addConstr(loading_sum >= min_loading, name=f"min_loading[{pos_id}]")
                model.addConstr(loading_sum <= max_loading, name=f"max_loading[{pos_id}]")
            
            # Discharging ports
            trade_discharging_ports = set(data['port_classifications']['discharging_ports'])
            discharging_ports_in_lane = list(trade_discharging_ports & lane_ports)
            discharging_sum = gp.quicksum(x[pos_id, p] for p in discharging_ports_in_lane)
            
            if discharging_ports_in_lane:
                model.addConstr(discharging_sum >= min_discharging, name=f"min_discharging[{pos_id}]")
                model.addConstr(discharging_sum <= max_discharging, name=f"max_discharging[{pos_id}]")


    # (C10) Maximum transit time for served commitments
    if ENFORCE_MAX_TRANSIT:
        transit_times = data['transit_times']
        count_c10 = 0
        for commit in commitments_list:
            cid = commit['id']
            from_port = commit['from']
            to_port = commit['to']
            if (from_port, to_port) in transit_times:
                max_transit = transit_times[(from_port, to_port)]
                for pos_id in pos_ids:
                    if (pos_id, cid) in z:
                        if (pos_id, from_port) in alpha and (pos_id, to_port) in alpha:
                            model.addConstr(
                                alpha[pos_id, to_port] - alpha[pos_id, from_port] 
                                <= max_transit + MAX_ROUTE_DURATION * (1 - z[pos_id, cid]),
                                name=f"max_transit[{pos_id},{from_port}->{to_port}]"
                            )
                            count_c10 += 1
        if verbose_level > 0:
            print(f"Added {count_c10} C10 max transit time constraints")


    model.update()

    # Objective and solve: minimize total time span (end - start) per position
    model.setObjective(
        gp.quicksum(
            lane_graphs[pos_info["lane_idx"]]["arc_times"][arc] * e[(pos_id, *arc)]
            for pos_info in positions
            for pos_id in [pos_info["id"]]
            for arc in lane_graphs[pos_info["lane_idx"]]["arc_times"]
            if (pos_id, *arc) in e
        ),
        GRB.MINIMIZE
    )

    status_labels = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
    }
    if verbose_level > 0:
        print("\nOptimizing model...\n")
    model.optimize()

    if model.SolCount is False:
        print("No feasible solution found; skipping route output.")
    

    # Store metadata
    model._meta = {
        "trade": trade_code,
        "positions": positions,
        "lane_graphs": lane_graphs,
        "source": SOURCE_NODE,
        "sink": SINK_NODE,
    }

    # Print clean summary if verbose_level < 2
    if model.SolCount and verbose_level < 2:
        print(f"\n{'='*70}")
        print(f"TRADE: {trade_code}")
        print(f"STATUS: OPTIMAL")
        print(f"OBJECTIVE: {model.ObjVal:.3f}")
        print(f"{'='*70}")
        
        for pos_info in positions:
            pos_id = pos_info["id"]
            lane_name = pos_info["lane_name"]
            ports = lane_graphs[pos_info["lane_idx"]]["ports"]
            visits = []
            for port in ports:
                if x[pos_id, port].X > 0.5:
                    visits.append((alpha[pos_id, port].X, port, phi[pos_id, port].X))
            visits.sort(key=lambda item: item[0])
            
            print(f"\nPosition {pos_id} ({lane_name}):")
            print(f"  Start Day: {s[pos_id].X:.0f}")
            print(f"  Port Sequence:")
            for i, (time_since_start, port, day_of_month) in enumerate(visits):
                # Calculate travel time to next port (difference in day_of_month)
                # Inside your loop over 'visits'
                if i < len(visits) - 1:
                    next_alpha = visits[i+1][0] # [0] is time_since_start
                    travel_time = next_alpha - time_since_start
                else:
                    travel_time = 0.0
                print(f"    {port:<30} Day-of-Month: {day_of_month+1:5.2f}  Time-since-start: {time_since_start:6.2f}d  Travel-time: {travel_time:5.2f}d")

    # Show final sequences when solution available
    if model.SolCount and verbose_level > 1:
        print(f"Objective value: {model.ObjVal:.3f}")
        print(f"Solver Runtime: {model.Runtime:.2f}s")
        print(f"MIP Gap: {model.MIPGap * 100:.4f}%")
        print("\n")
        print(f"{'='*70}")
        for pos_info in positions:
            pos_id = pos_info["id"]
            lane_name = pos_info["lane_name"]
            ports = lane_graphs[pos_info["lane_idx"]]["ports"]
            visits = []
            for port in ports:
                if x[pos_id, port].X > 0.5:
                    visits.append((alpha[pos_id, port].X, port, phi[pos_id, port].X))
            visits.sort(key=lambda item: item[0])
            # Compute duration from selected arcs
            selected_arcs = [(i, j) for i, j in lane_graphs[pos_info["lane_idx"]]["arcs"] if e.get((pos_id, i, j), 0).X > 0.5]
            duration = sum(lane_graphs[pos_info["lane_idx"]]["arc_times"].get(arc, 0.0) for arc in selected_arcs)
            print(f"\nPosition {pos_id} (lane {lane_name}):")
            print(f"  Start day: s={s[pos_id].X:.0f}")
            print(f"  Duration: {duration:.3f} days")
            # Count loading/discharging ports visited in this position
            visited_ports = [port for _, port, _ in visits]
            trade_loading = set(data['port_classifications']['loading_ports'])
            trade_discharging = set(data['port_classifications']['discharging_ports'])
            num_loading_visited = len(set(visited_ports) & trade_loading)
            num_discharging_visited = len(set(visited_ports) & trade_discharging)
            print(f"  Loading ports visited: {num_loading_visited}")
            print(f"  Discharging ports visited: {num_discharging_visited}")
            for order, (time_since_start, port, day_of_month) in enumerate(visits, 1):
                # Calculate travel time to next port (difference in day_of_month)
                # Inside your loop over 'visits'
                if i < len(visits) - 1:
                    next_alpha = visits[i+1][0] # [0] is time_since_start
                    travel_time = next_alpha - time_since_start
                else:
                    travel_time = 0.0
                
                print(f"  #{order:02d} {port:<25} Day-of-Month: {day_of_month+1:5.2f}  Time-since-start: {time_since_start:6.2f}d  Travel-time: {travel_time:5.2f}d")
            
            # Debug: verify timing accuracy (show travel times between consecutive ports)
            
            if visits and verbose_level > 0:
                # Check first port (should have alpha = 0, since route starts there)
                print(f"  Travel times verification:")
                first_port = visits[0][1]
                first_alpha = visits[0][0]
                expected_first = 0.0  # First port arrival time should always be 0
                diff = abs(first_alpha - expected_first)
                status = "✓" if diff < 0.01 else "✗ MISMATCH"
                print(f"    SOURCE → {first_port}: alpha={first_alpha:.3f}d, expected=0.000d {status}")
            

            for i in range(len(visits)-1):
                port_i = visits[i][1]
                port_j = visits[i+1][1]
                alpha_i = visits[i][0]
                alpha_j = visits[i+1][0]
                actual_time = alpha_j - alpha_i
                # Get expected time from arc_times
                expected_time = lane_graphs[pos_info["lane_idx"]]["arc_times"].get((port_i, port_j), None)
                if expected_time is not None:
                    diff = abs(actual_time - expected_time)
                    status = "✓" if diff < 0.01 else "✗ MISMATCH"
                    print(f"    {port_i} → {port_j}: actual={actual_time:.3f}d, expected={expected_time:.3f}d {status}")
                else:
                    print(f"    {port_i} → {port_j}: actual={actual_time:.3f}d (no expected time found)")
            # C1: Commitments served by this position
            print(f"  Commitments served by this position:")
            served_commits = 0
            for commit in commitments_list:
                cid = commit["id"]
                if (pos_id, cid) in z and z[pos_id, cid].X > 0.5:
                    from_port = commit["from"]
                    to_port = commit["to"]
                    transit_time = alpha[pos_id, to_port].X - alpha[pos_id, from_port].X
                    print(f"    - {from_port} -> {to_port} (Transit: {transit_time:.2f}d)")
                    served_commits += 1
            if served_commits == 0:
                print("    (None)")
        print("\n" + "="*60)
        print("CONSTRAINT COMPLIANCE REPORT")
        print("="*60)
        
        
        # C2: Start Spread
        if ENFORCE_START_SPREAD and num_positions > 1:
            min_spread = data['min_spread']
            print(f"\nC3 Start Spread Constraint:")
            print(f"  Required minimum: {min_spread} days")
            sorted_pos = sorted(pos_ids, key=lambda p: s[p].X)
            for p in sorted_pos:
                print(f"    {p}: s={s[p].X:.0f}")
            if len(sorted_pos) > 1:
                spreads = [s[sorted_pos[i+1]].X - s[sorted_pos[i]].X for i in range(len(sorted_pos)-1)]
                min_actual = min(spreads)
                print(f"  Actual minimum spread: {min_actual:.1f} days")
                status = "OK" if min_actual >= min_spread - 0.01 else "VIOLATED"
                print(f"  Status: {status}")
        
        # C4: Port Spread
        if ENFORCE_PORT_SPREAD:
            ports_with_constraints = data['ports_with_constraints']
            print(f"\nC4 Port Spread Constraints:")
            has_violations = False
            for port in ports_with_constraints:
                min_sp = ports_with_constraints[port]['min_spread']
                if min_sp > 0:
                    visits = [(p, phi[p, port].X) for p in pos_ids if (p, port) in phi and x[p, port].X > 0.5]
                    if len(visits) >= 2:
                        visits.sort(key=lambda v: v[1])
                        actual = min(visits[i+1][1] - visits[i][1] for i in range(len(visits)-1))
                        status = "OK" if actual >= min_sp - 0.01 else "VIOLATED"
                        if status == "VIOLATED":
                            has_violations = True
                        print(f"  {port}: required={min_sp}, actual={actual:.1f} ({status})")
            if not has_violations:
                print("  All port spread constraints satisfied")
        
        # C5: Start Range (At Least One Position)
        if ENFORCE_START_RANGE and data.get('route_start_range'):
            print(f"\nC5 Start Range (At Least One Position):")
            lower, upper = data['route_start_range']['lower'], data['route_start_range']['upper']
            print(f"  Target range: [{lower}, {upper}]")
            designated_positions = [pos_id for pos_id in pos_ids if y_start_range[pos_id].X > 0.5]
            if designated_positions:
                print(f"  Designated positions in range:")
                all_compliant = True
                for pos_id in designated_positions:
                    s_val = s[pos_id].X
                    compliant = lower <= s_val <= upper + 0.01
                    status = "OK" if compliant else "VIOLATED"
                    if not compliant:
                        all_compliant = False
                    print(f"    {pos_id}: s={s_val:.0f} ({status})")
                overall_status = "OK" if all_compliant and len(designated_positions) >= 1 else "VIOLATED"
                print(f"  Overall status: {overall_status}")
            else:
                print(f"  WARNING: No positions designated (should have at least 1)")
                print(f"  Status: VIOLATED")
        
        # C6: Death Position Avoidance
        if ENFORCE_DEATH_AVOIDANCE and data.get('route_dead_position'):
            print(f"\nC6 Death Position Avoidance:")
            dead_l, dead_u = data['route_dead_position']['lower'], data['route_dead_position']['upper']
            print(f"  Death range: [{dead_l}, {dead_u}]")
            all_safe = True
            for pos_id in pos_ids:
                s_val = s[pos_id].X
                if s_val <= dead_l - 1:
                    status = "BELOW (safe)"
                elif s_val >= dead_u + 1:
                    status = "ABOVE (safe)"
                else:
                    status = "INSIDE (VIOLATED)"
                    all_safe = False
                print(f"    {pos_id}: s={s_val:.0f} ({status})")
            if all_safe:
                print("  All positions avoid death range")
        
        # C8: Port Visit Windows (At Least One Position Per Port)
        if ENFORCE_PORT_WINDOWS:
            ports_with_constraints = data['ports_with_constraints']
            print(f"\nC8 Port Visit Windows (At Least One Position Per Port):")
            has_violations = False
            for port in ports_with_constraints:
                lower = ports_with_constraints[port]['desired_position_lower']
                upper = ports_with_constraints[port]['desired_position_upper']
                if lower > 0 or upper > 0:
                    # Find positions that visit this port
                    visiting_positions = [(p, phi[p, port].X) for p in pos_ids 
                                        if (p, port) in phi and x[p, port].X > 0.5]
                    if visiting_positions:
                        print(f"  {port}: window=[{lower}, {upper}]")
                        # Find designated positions (should be at least 1)
                        designated = [p for p in pos_ids if (p, port) in y_port_window 
                                    and y_port_window[p, port].X > 0.5]
                        
                        if not designated:
                            print(f"    WARNING: No position designated for window (should have >= 1)")
                            has_violations = True
                        
                        for pos_id, phi_val in visiting_positions:
                            is_designated = pos_id in designated
                            in_window = lower <= (phi_val+1) <= upper + 0.01
                            
                            if is_designated:
                                status = "DESIGNATED, OK" if in_window else "DESIGNATED, VIOLATED"
                                if not in_window:
                                    has_violations = True
                            else:
                                status = "not designated (can be outside)"
                            
                            print(f"    {pos_id}: phi={phi_val+1:.1f} ({status})")
            if not has_violations:
                print("  All port window constraints satisfied")
        
        # C11: Maximum Transit Time
        if ENFORCE_MAX_TRANSIT:
            transit_times = data['transit_times']
            print(f"\nC11 Maximum Transit Time:")
            has_violations = False
            for commit in commitments_list:
                cid, from_p, to_p = commit['id'], commit['from'], commit['to']
                if (from_p, to_p) in transit_times:
                    max_t = transit_times[(from_p, to_p)]
                    served_positions = [p for p in pos_ids if (p, cid) in z and z[p, cid].X > 0.5]
                    if served_positions:
                        print(f"  {from_p} -> {to_p}: max={max_t} days")
                        for pos_id in served_positions:
                            if (pos_id, from_p) in alpha and (pos_id, to_p) in alpha:
                                actual = alpha[pos_id, to_p].X - alpha[pos_id, from_p].X
                                status = "OK" if actual <= max_t + 0.01 else "VIOLATED"
                                if status == "VIOLATED":
                                    has_violations = True
                                print(f"      {pos_id}: {actual:.1f} days ({status})")
            if not has_violations:
                print("  All transit time constraints satisfied")
    
    # Extract results for return
    result = {
        'status': status_labels.get(model.Status, str(model.Status)),
        'objective': model.ObjVal if model.SolCount else None,
        'runtime': model.Runtime if hasattr(model, 'Runtime') else None,
        'routes': []
    }
    
    if model.SolCount:
        for pos_info in positions:
            pos_id = pos_info["id"]
            lane_name = pos_info["lane_name"]
            ports = lane_graphs[pos_info["lane_idx"]]["ports"]
            visits_list = []
            for port in ports:
                if x[pos_id, port].X > 0.5:
                    visits_list.append({
                        'port': port,
                        'arrival_day': alpha[pos_id, port].X,
                        'day_of_month': phi[pos_id, port].X+1
                    })
            visits_list.sort(key=lambda v: v['arrival_day'])
            
            result['routes'].append({
                'position_id': pos_id,
                'lane_name': lane_name,
                'start_day': s[pos_id].X,
                'visits': visits_list
            })
    
    return result


# Main execution block for backward compatibility
if __name__ == "__main__":
    import sys
    trade_code = sys.argv[1] if len(sys.argv) > 1 else "EUAF"
    verbose_level = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    build_and_solve_mip(trade_code, verbose_level=verbose_level)
