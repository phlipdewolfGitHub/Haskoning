import pandas as pd
from pathlib import Path

# Control print verbosity
verbose = True  # Set to False to suppress detailed output

# Helper function to print only when verbose is True
def vprint(*args, **kwargs):
    if verbose:
        print(*args, **kwargs)

base_dir = Path(__file__).parent

# ================================
# Commitments analysis
# ================================
commitments_path = base_dir / 'Data/PriorityCommitments.csv'

df = pd.read_csv(commitments_path, sep=';')
vprint(f"Columns: {list(df.columns)}")

df.columns = df.columns.str.strip()
df = df.dropna(subset=[df.columns[0]])
trade_col = df.columns[0]
port_from_col = df.columns[1]
port_to_col = df.columns[2]
value_col = df.columns[3]

vprint("\n" + "="*50)
vprint("COMMITMENTS PER TRADE")
vprint("="*50)
# Sum commitments per trade
df[value_col] = pd.to_numeric(df[value_col], errors='coerce').fillna(0)
trade_commitments = df.groupby(trade_col)[value_col].sum().sort_values(ascending=False)
vprint(trade_commitments)
vprint(f"\nTotal trades: {len(trade_commitments)}")
vprint(f"Total commitments (rows): {len(df)}")

vprint("\n" + "="*50)
vprint("OUTGOING COMMITMENTS PER PORT")
vprint("="*50)
outgoing_commitments = df.groupby(port_from_col)[value_col].sum().sort_values(ascending=False)
vprint("Top 10 ports by outgoing commitments:")
vprint(outgoing_commitments.head(10))
vprint(f"\nTotal unique departure ports: {len(outgoing_commitments)}")

vprint("\n" + "="*50)
vprint("INCOMING COMMITMENTS PER PORT")
vprint("="*50)
incoming_commitments = df.groupby(port_to_col)[value_col].sum().sort_values(ascending=False)
vprint("Top 10 ports by incoming commitments:")
vprint(incoming_commitments.head(10))
vprint(f"\nTotal unique destination ports: {len(incoming_commitments)}")

vprint("\n" + "="*50)
vprint("TOTAL PORT ACTIVITY (INCOMING + OUTGOING COMMITMENTS)")
vprint("="*50)
# Combine outgoing and incoming commitments
port_activity = pd.DataFrame({
    'Outgoing': outgoing_commitments,
    'Incoming': incoming_commitments
}).fillna(0)
port_activity['Total'] = port_activity['Outgoing'] + port_activity['Incoming']
port_activity = port_activity.sort_values('Total', ascending=False)
vprint("Top 15 most active ports (by total commitments):")
vprint(port_activity.head(15))

vprint("\n" + "="*50)
vprint("PORTS PARTICIPATING IN MULTIPLE TRADES")
vprint("="*50)

# Consider only positive commitments
df_pos = df[df[value_col] > 0].copy()

# Stack origin and destination ports into a single column with their trade
port_trade = pd.concat(
    [
        df_pos[[trade_col, port_from_col]].rename(columns={port_from_col: "Port"}),
        df_pos[[trade_col, port_to_col]].rename(columns={port_to_col: "Port"}),
    ],
    ignore_index=True,
)

# Unique trade count per port
trade_counts_by_port = port_trade.groupby("Port")[trade_col].nunique()

# Ports in more than one trade
ports_multi_trade = trade_counts_by_port[trade_counts_by_port > 1].sort_values(ascending=False)

vprint(f"Total ports in more than one trade: {len(ports_multi_trade)}")

# Map ports to sorted list of trades
trades_by_port = port_trade.groupby("Port")[trade_col].apply(lambda s: sorted(s.unique()))

# Print ports and their trades
for port in ports_multi_trade.index:
    vprint(f"- {port}: {', '.join(trades_by_port[port])}")

vprint("\n" + "="*50)
vprint("PER-PORT, PER-TRADE LOADING/DISCHARGING SUMMARY")
vprint("="*50)

# Positive commitments only and normalize whitespace
df_pos_comm = df[df[value_col] > 0].copy()
for col in [trade_col, port_from_col, port_to_col]:
    df_pos_comm[col] = df_pos_comm[col].astype(str).str.strip()

# Outgoing (loading) per trade/port
out = (
    df_pos_comm.groupby([trade_col, port_from_col])[value_col]
    .sum()
    .rename("Loading")
    .reset_index()
    .rename(columns={port_from_col: "Port"})
)

# Incoming (discharging) per trade/port
inc = (
    df_pos_comm.groupby([trade_col, port_to_col])[value_col]
    .sum()
    .rename("Discharging")
    .reset_index()
    .rename(columns={port_to_col: "Port"})
)

# Combine and classify
summary = pd.merge(out, inc, on=[trade_col, "Port"], how="outer").fillna(0)

def classify_port(row):
    has_out = row["Loading"] > 0
    has_in = row["Discharging"] > 0
    if has_out and has_in:
        return "Co-loading"
    if has_out:
        return "Loading"
    if has_in:
        return "Discharging"
    return "No activity"

summary["Type"] = summary.apply(classify_port, axis=1)

# Print every port and per-trade breakdown
for port in sorted(summary["Port"].unique()):
    sub = summary[summary["Port"] == port].sort_values(trade_col)
    vprint(f"\nPort: {port}")
    for _, r in sub.iterrows():
        vprint(f"  - {r[trade_col]} | Loading={int(r['Loading'])} | Discharging={int(r['Discharging'])} | Type={r['Type']}")

vprint("\n" + "="*50)
vprint("SUMMARY STATISTICS")
vprint("="*50)
vprint(f"Total unique ports: {len(set(df[port_from_col].unique()) | set(df[port_to_col].unique()))}")
total_commitments = df[value_col].sum()
vprint(f"Total commitments (sum of all commitments): {total_commitments}")

vprint("\n" + "="*50)
vprint("ANALYSIS COMPLETE")
vprint("="*50)

# =========================================================
# Distances: normal parse, round to 1 decimal, top-10 extremes
# =========================================================
dist_path = base_dir / "Data/Distances.csv"
dfd = pd.read_csv(dist_path)  # expects columns: Port from, Port to, distance

# Exclude zero distances for closest calculation
df_pos = dfd[dfd["distance"] > 0].copy()

closest10 = df_pos.nsmallest(10, "distance")[["Port from", "Port to", "distance"]]
furthest10 = df_pos.nlargest(10, "distance")[["Port from", "Port to", "distance"]]

vprint("\n10 closest non-zero routes:")
vprint(closest10.to_string(index=False))

vprint("\n10 furthest routes:")
vprint(furthest10.to_string(index=False))

vprint("\n" + "="*50)
vprint("PORT-PORT PAIR ANALYSIS")
vprint("="*50)

# Create a dictionary of port pairs with their trade information and distances
port_pairs = {}

# First, build the distance lookup dictionary
distance_dict = {}
for _, row in dfd.iterrows():
    key = (row["Port from"].strip(), row["Port to"].strip())
    distance_dict[key] = row["distance"]

# Now build port pair information from commitments
for _, row in df.iterrows():
    port_from = row[port_from_col].strip()
    port_to = row[port_to_col].strip()
    trade = row[trade_col].strip()
    commitments = int(row[value_col]) if pd.notna(row[value_col]) else 0
    
    # Skip zero commitment entries
    if commitments <= 0:
        continue
    
    # Create key for the pair (ensure consistent ordering)
    pair_key = (port_from, port_to)
    
    if pair_key not in port_pairs:
        port_pairs[pair_key] = {
            'trades': {},
            'distance': distance_dict.get(pair_key, None)
        }
    
    if trade not in port_pairs[pair_key]['trades']:
        port_pairs[pair_key]['trades'][trade] = 0
    
    port_pairs[pair_key]['trades'][trade] += commitments

vprint(f"Total port pairs with commitments: {len(port_pairs)}")

# Print some sample port pairs to verify
vprint("\nSample port pairs (first 5):")
count = 0
for pair_key, pair_data in port_pairs.items():
    if count >= 5:
        break
    vprint(f"\nPort pair: {pair_key[0]} -> {pair_key[1]}")
    vprint(f"  Distance: {pair_data['distance'] if pair_data['distance'] is not None else 'Not found'}")
    vprint(f"  Trades: {list(pair_data['trades'].keys())}")
    for trade, commitments in pair_data['trades'].items():
        vprint(f"    {trade}: {commitments} commitments")
    count += 1

# Print summary statistics for port pairs
total_commitments_in_pairs = sum(sum(pair_data['trades'].values()) for pair_data in port_pairs.values())
vprint(f"\nTotal commitments in port pairs: {total_commitments_in_pairs}")
vprint(f"Average commitments per port pair: {total_commitments_in_pairs / len(port_pairs):.2f}")
vprint(f"Average distance between port pairs: {sum(pair_data['distance'] for pair_data in port_pairs.values() if pair_data['distance'] is not None) / len([p for p in port_pairs.values() if p['distance'] is not None]):.1f}")

vprint("\n" + "="*50)
vprint("PORT-PAIR ANALYSIS COMPLETE")
vprint("="*50)

vprint("\n" + "="*50)
vprint("PORT PAIRS IN MULTIPLE TRADES")
vprint("="*50)
multi_trade_pairs = [(pair, data) for pair, data in port_pairs.items() if len(data['trades']) > 1]
vprint(f"Total port pairs across multiple trades: {len(multi_trade_pairs)}")

# Show a few examples
for pair, data in multi_trade_pairs[:10]:
    vprint(f"- {pair[0]} -> {pair[1]} | Trades: " + ", ".join(f"{t}({c})" for t, c in data['trades'].items()) + f" | Distance: {data['distance']}")

vprint("\n" + "="*50)
vprint("COMMITMENT VALUE FREQUENCY PER TRADE")
vprint("="*50)
df_freq = df.copy()
df_freq[value_col] = pd.to_numeric(df_freq[value_col], errors="coerce")
# Count how many rows per trade have commitment value 1,2,3,...
freq = (
    df_freq.groupby(trade_col)[value_col]
    .value_counts(dropna=False)
    .sort_index()
    .rename("frequency")
    .reset_index()
)

# Print per-trade distributions
for trade in freq[trade_col].unique():
    sub = freq[freq[trade_col] == trade][[value_col, "frequency"]]
    vprint(f"\nTrade: {trade}")
    vprint(sub.to_string(index=False))
# Always print this regardless of verbose setting
print(f"Analysis complete. Created {len(port_pairs)} port pairs with {total_commitments_in_pairs} total commitments.")