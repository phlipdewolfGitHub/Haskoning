"""
Batch runner for optimizing multiple trades.
Provides clean, concise output showing only routes, dates, and objective values.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from same directory
from MIP_test import build_and_solve_mip

# All available trades
ALL_TRADES = ["EUAF", "EUME", "FEEU", "FEUS", "MIAF", "USME"]

# Output verbosity level: 0=minimal, 1=basic, 2=detailed
VERBOSE_LEVEL = 0


def print_route_summary(trade_code, result):
    """Print clean summary of routes for a trade."""
    if not result or result['status'] != 'OPTIMAL':
        status = result.get('status', 'UNKNOWN') if result else 'NO_RESULT'
        print(f"\n{'='*70}")
        print(f"TRADE: {trade_code}")
        print(f"STATUS: {status}")
        if result and 'objective' in result:
            print(f"OBJECTIVE: {result.get('objective', 'N/A')}")
        print(f"{'='*70}")
        return
    
    print(f"\n{'='*70}")
    print(f"TRADE: {trade_code}")
    print(f"STATUS: {result['status']}")
    print(f"OBJECTIVE: {result['objective']:.3f}")
    print(f"{'='*70}")
    
    for route in result['routes']:
        pos_id = route['position_id']
        lane_name = route['lane_name']
        start_day = route['start_day']
        
        print(f"\nPosition {pos_id} ({lane_name}):")
        print(f"  Start Day: {start_day:.0f}")
        print(f"  Port Sequence:")
        
        for visit in route['visits']:
            port = visit['port']
            arrival_day = visit['arrival_day']
            day_of_month = visit['day_of_month']
            print(f"    {port:<30} Arrival: {arrival_day:6.2f}d  Day-of-Month: {day_of_month:5.2f}")


def run_trades(trade_codes=None, verbose_level=None):
    """
    Run optimization for multiple trades.
    
    Args:
        trade_codes: List of trade codes to run. If None, runs all trades.
        verbose_level: Output verbosity (0=minimal, 1=basic, 2=detailed). If None, uses VERBOSE_LEVEL.
    
    Returns:
        Dictionary mapping trade_code -> result dictionary
    """
    if trade_codes is None:
        trade_codes = ALL_TRADES
    
    if verbose_level is None:
        verbose_level = VERBOSE_LEVEL
    
    results = {}
    
    for trade_code in trade_codes:
        print(f"\n{'#'*70}")
        print(f"Processing trade: {trade_code}")
        print(f"{'#'*70}")
        
        try:
            result = build_and_solve_mip(trade_code, verbose_level=verbose_level)
            results[trade_code] = result
            
            if verbose_level == 0:
                # Minimal mode: print only summary
                print_route_summary(trade_code, result)
                
        except Exception as e:
            print(f"ERROR processing {trade_code}: {e}")
            results[trade_code] = {
                'status': 'ERROR',
                'error': str(e)
            }
    
    return results


if __name__ == "__main__":
    import argparse
    from datetime import datetime
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description='Run MIP optimization for maritime trades.\n\n'
    )
    parser.add_argument('--trades', nargs='+', choices=ALL_TRADES + ['ALL'],
                        default=['ALL'], help='Trade codes to run (default: ALL)')
    parser.add_argument('--verbose-level', '-v', type=int, default=None, choices=[0, 1, 2],
                        help='Output verbosity: 0=minimal, 1=basic, 2=detailed (default: VERBOSE_LEVEL constant)')
    parser.add_argument('--output-file', '-o', type=str, default=None,
                        help='File name for output, stored in Network_Scheduling/RESULTS/ (default: <timestamp>Failed.txt)')

    args = parser.parse_args()

    if 'ALL' in args.trades:
        trade_codes = ALL_TRADES
    else:
        trade_codes = args.trades

    # Set verbose level to 2 if not specified
    verbose_level = args.verbose_level if args.verbose_level is not None else 2

    # Output path logic: always in Network_Scheduling/RESULTS, which is one dir up from this script
    script_dir = Path(__file__).parent
    results_dir = (script_dir.parent / "RESULTS").resolve()

    if args.output_file:
        output_filename = args.output_file
        output_path = results_dir / output_filename
    else:
        print(f"No output file specified, using default: {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}Failed.txt")
        output_path = results_dir / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}Failed.txt"

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Redirect output to file
    with open(output_path, 'w') as f:
        # Redirect stdout to file
        import sys
        original_stdout = sys.stdout
        sys.stdout = f
        
        try:
            print(f"Running all trades at verbose level {verbose_level}")
            print(f"Output file: {output_path}")
            print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*70}\n")
            
            results = run_trades(trade_codes, verbose_level=verbose_level)
            
            # Summary statistics
            print(f"\n{'='*70}")
            print("SUMMARY")
            print(f"{'='*70}")
            optimal_count = sum(1 for r in results.values() if r.get('status') == 'OPTIMAL')
            print(f"Trades processed: {len(results)}")
            print(f"Optimal solutions: {optimal_count}")
            print(f"Failed/Infeasible: {len(results) - optimal_count}")
            
            # Calculate total duration across all optimal solutions
            # The objective value from each trade is the total duration for that trade
            total_duration = 0.0
            for trade_code, result in results.items():
                if result.get('status') == 'OPTIMAL' and result.get('objective') is not None:
                    print(f"\n  {trade_code}: {result['objective']:.2f} days")
                    total_duration += result['objective']
            
            print(f"\n{'='*70}")
            print("CAPACITY ANALYSIS")
            print(f"{'='*70}")
            print(f"Total combined duration of all routes: {total_duration:.2f} days")
            estimated_vessels = total_duration / 30  # CYCLE_DAYS = 30
            print(f"Estimated number of vessels needed (total_duration / 30): {estimated_vessels:.2f}")
            print(f"{'='*70}")
            
            # Runtime table
            print(f"\n{'='*70}")
            print("RUNTIME ANALYSIS")
            print(f"{'='*70}")
            print(f"{'Trade':<10} {'Runtime (s)':<15}")
            print(f"{'-'*25}")
            total_runtime = 0.0
            for trade_code in trade_codes:
                result = results.get(trade_code, {})
                runtime = result.get('runtime')
                if runtime is not None:
                    print(f"{trade_code:<10} {runtime:>14.2f}")
                    total_runtime += runtime
                else:
                    print(f"{trade_code:<10} {'N/A':>14}")
            print(f"{'-'*25}")
            print(f"{'TOTAL':<10} {total_runtime:>14.2f}")
            print(f"{'='*70}")
            print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
        finally:
            # Restore stdout
            sys.stdout = original_stdout
    
    print(f"Results written to: {output_path}")

