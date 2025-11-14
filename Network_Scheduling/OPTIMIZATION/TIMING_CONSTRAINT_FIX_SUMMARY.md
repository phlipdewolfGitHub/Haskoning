# Maritime Route Scheduling MIP Model - Timing Constraint Fix Summary

## Project Overview

This is a **Mixed Integer Programming (MIP) model** built with Gurobi for solving a **maritime vessel scheduling problem**. The model optimizes routes for multiple vessel positions (vessels) operating within a trade corridor, determining:

- **Which ports each vessel visits** (binary variable `x[pos_id, port]`)
- **The sequence of port visits** (binary variable `e[pos_id, i, j]` for arcs)
- **When each route starts** (continuous variable `s[pos_id]` - start day in the 30-day cycle)
- **When each port is reached** (continuous variable `alpha[pos_id, port]` - arrival time in days from route start)
- **Day of month for each port visit** (continuous variable `phi[pos_id, port]` - computed via modulo: `phi = (s + alpha) mod 30`)
- **Which commitments are served** (binary variable `z[pos_id, commit_id]`)

### Key Variables

- **`s[pos_id]`**: Route start day (0-29, representing day of month)
- **`alpha[pos_id, port]`**: Arrival time at port (in days from route start)
- **`phi[pos_id, port]`**: Day of month when port is visited (`phi = s + alpha mod 30`)
- **`e[pos_id, i, j]`**: Binary indicator if arc from port `i` to port `j` is used
- **`tau[(i,j)]`**: Travel time from port `i` to port `j` (loaded from distance matrix, includes sailing + port time)

### Model Structure

- **Trade corridors** contain multiple **lanes** (different route types)
- Each lane has multiple **positions** (vessels)
- Each lane has a **DAG structure** with SOURCE → ports → SINK
- Travel times (`tau`) are precomputed from distance matrices (include sailing time + port handling time)

## The Problem We Fixed

### Issue Identified

The model had **three critical timing problems**:

1. **First port arrival time was incorrect**: The first port's `alpha` value was incorrectly set to `port_time` (which could be > 0) instead of 0. Since the route starts on day `s` and immediately goes to the first port (no travel time from SOURCE), the first port should have `alpha = 0` exactly. This ensures arrival happens on day `s` of the route start.

2. **Travel time slack**: The model was allowing **flexibility/slack** in travel times between ports, meaning `alpha[j] - alpha[i]` could be greater than the exact `tau[(i,j)]` value from the distance matrix. This violated the requirement to use exact travel times.

3. **Inconsistent first port timing across positions**: The first position might work correctly (if `port_time` happened to be 0), but other positions showed incorrect `alpha > 0` for their first port, causing inconsistent behavior.

### Root Cause

**Issue 1 & 2 (Travel time slack)**: The timing recursion constraint was implemented as a **single inequality**:

```python
alpha[pos_id, j] >= alpha[pos_id, i] + tau_arc - M_time * (1 - e[pos_id, i, j])
```

This only enforced a **lower bound**, allowing `alpha[j]` to be arbitrarily larger than `alpha[i] + tau` when the arc was selected. This created unintended slack in travel times.

**Issue 3 (First port timing)**: For SOURCE → port arcs, the constraint enforced `alpha[port] = tau_arc`, where `tau_arc` was set to `port_time` (which could be > 0). This was conceptually incorrect because:
- The route starts on day `s` and immediately goes to the first port (no travel time from SOURCE)
- The first port should have `alpha = 0` exactly
- Port handling time is accounted for in the port visit itself, not in the arrival time
- This caused inconsistent behavior: positions with `port_time = 0` appeared correct, while others showed `alpha > 0` incorrectly

## The Exact Fix

### Change 1: Enforce Exact Equality for Port-to-Port Travel

**Before** (single inequality):
```python
# Only lower bound - allows slack
model.addConstr(
    alpha[pos_id, j] >= alpha[pos_id, i] + tau_arc - M_time * (1 - e[pos_id, i, j]),
    name=f"timing[{pos_id},{i},{j}]"
)
```

**After** (dual inequalities):
```python
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
```

**How it works**: When `e[pos_id, i, j] = 1` (arc selected), both inequalities combine to enforce `alpha[j] = alpha[i] + tau` exactly. When `e = 0`, the Big-M terms (`M_time`) make both constraints inactive.

### Change 2: Enforce Exact First Port Arrival Time (alpha = 0)

**Before** (enforced alpha = tau_arc, which could be > 0):
```python
# Lower bound - enforced alpha = port_time (could be > 0)
model.addConstr(alpha[pos_id, j] >= tau_arc * e[pos_id, i, j],
              name=f"alpha_init_lb[{pos_id},{j}]")
# Upper bound - enforced alpha = port_time
model.addConstr(alpha[pos_id, j] <= tau_arc * e[pos_id, i, j] + M_time * (1 - e[pos_id, i, j]),
              name=f"alpha_init_ub[{pos_id},{j}]")
```

**After** (enforces alpha = 0 exactly):
```python
# First port: alpha must be exactly 0 when arc is selected
# The route starts on day s, and we immediately go to the first port
# Port handling time is accounted for in the port visit, not in arrival time
# Lower bound: alpha >= 0 when e=1, or >= -M when e=0
model.addConstr(alpha[pos_id, j] >= 0 - M_time * (1 - e[pos_id, i, j]),
              name=f"alpha_init_lb[{pos_id},{j}]")
# Upper bound: alpha <= 0 when e=1, or <= M when e=0
model.addConstr(alpha[pos_id, j] <= 0 + M_time * (1 - e[pos_id, i, j]),
              name=f"alpha_init_ub[{pos_id},{j}]")
```

**How it works**: When `e[pos_id, SOURCE, j] = 1` (first port selected), both inequalities enforce `alpha[j] = 0` exactly. This is correct because:
- The route starts on day `s`
- We immediately go to the first port (no travel time from SOURCE)
- Arrival time should be 0 (we arrive on the start day)
- Port handling time is accounted for in the port visit itself, not in the arrival time

**Why this matters**: Previously, if `tau_arc = port_time > 0`, the first port would have `alpha > 0`, meaning it would arrive on day `s + port_time` instead of day `s`. This caused incorrect day-of-month calculations (`phi = s + alpha` would be wrong).

### Change 3: Added Verification Output

Added detailed verification in the solution output to confirm timing accuracy:

```python
# Travel times verification section
print(f"  Travel times verification:")
# Check first port (should have alpha = 0, since route starts there)
first_port = visits[0][1]
first_alpha = visits[0][0]
expected_first = 0.0  # First port arrival time should always be 0
diff = abs(first_alpha - expected_first)
status = "✓" if diff < 0.01 else "✗ MISMATCH"
print(f"    SOURCE → {first_port}: alpha={first_alpha:.3f}d, expected=0.000d {status}")

# Check each port-to-port segment
for i in range(len(visits)-1):
    port_i = visits[i][1]
    port_j = visits[i+1][1]
    actual_time = alpha_j - alpha_i
    expected_time = lane_graphs[lane_idx]["arc_times"].get((port_i, port_j), None)
    if expected_time is not None:
        diff = abs(actual_time - expected_time)
        status = "✓" if diff < 0.01 else "✗ MISMATCH"
        print(f"    {port_i} → {port_j}: actual={actual_time:.3f}d, expected={expected_time:.3f}d {status}")
```

Also added start day display:
```python
print(f"  Start day: s={s[pos_id].X:.0f}")
```

## What to Look For in Results

### Expected Behavior

1. **First port `alpha` value**: Should be exactly **0** for all positions. The route starts on day `s` and immediately goes to the first port (no travel time from SOURCE), so `alpha[first_port] = 0` exactly.

2. **Port-to-port travel times**: For consecutive ports `i → j`, the difference `alpha[j] - alpha[i]` should exactly match `tau[(i,j)]` from the distance matrix (within 0.01 day tolerance)

3. **Day of month calculation**: `phi[pos_id, port] = (s[pos_id] + alpha[pos_id, port]) mod 30` should be correct. For the first port, `phi = s` exactly (since `alpha = 0`)

4. **Verification output**: All travel time checks should show **✓** (not ✗ MISMATCH). The first port check should show `alpha=0.000d, expected=0.000d ✓`

### Example Output Interpretation

```
Position (0,1) (lane FEUS_FEUS):
  Start day: s=25
  Duration: 45.250 days
  #01 Taicang                    arrival=0.000d  day_of_month=25.00
  #02 Masan                       arrival=2.725d  day_of_month=27.73
  Travel times verification:
    SOURCE → Taicang: alpha=0.000d, expected=0.000d ✓
    Taicang → Masan: actual=2.725d, expected=2.725d ✓
```

**Key checks**:
- First port `alpha` (0.000d) equals 0 exactly ✓
- Travel time Taicang → Masan (2.725d) matches expected from distance matrix ✓
- Day of month for first port (25.00) = start day (25) exactly (since `alpha = 0`) ✓
- All positions should show `alpha = 0` for their first port, ensuring consistent behavior ✓

### Red Flags

- **✗ MISMATCH** indicators in verification output
- First port `alpha` not equal to 0 (should be exactly 0.000d)
- First port `alpha` different across positions (all should be 0)
- Port-to-port travel times not matching distance matrix values
- Day of month for first port not equal to `s` (should be exactly `s` since `alpha = 0`)

## Technical Details

### Big-M Formulation

The fix uses **Big-M formulation** to conditionally enforce equality:
- When `e = 1`: Constraints become `alpha[j] >= alpha[i] + tau` AND `alpha[j] <= alpha[i] + tau` → equality
- When `e = 0`: Constraints become `alpha[j] >= -M` AND `alpha[j] <= M` → inactive (no restriction)

`M_time = MAX_ROUTE_DURATION = 200` days is used as the Big-M value.

### Constraint Count Impact

The fix **doubles** the number of timing constraints:
- **Before**: 1 constraint per arc (lower bound only)
- **After**: 2 constraints per arc (lower + upper bound)

This ensures exact travel times but increases model size slightly.

## Files Modified

- **`MIP_test.py`**: Lines 510-548 (timing recursion constraints), Lines 649 (start day display), Lines 662-687 (verification output)

## Status

✅ **Fixed and verified**: The model now enforces:
- **Exact first port arrival time**: `alpha[first_port] = 0` for all positions (consistent behavior)
- **Exact port-to-port travel times**: No slack allowed, `alpha[j] = alpha[i] + tau` exactly when arcs are selected
- **Correct day-of-month calculations**: First port visited on day `s` exactly (`phi = s` since `alpha = 0`)

Verification output confirms all timing constraints are satisfied with ✓ indicators.

