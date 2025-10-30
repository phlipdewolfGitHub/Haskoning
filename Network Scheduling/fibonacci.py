"""
Calculate and display the first 20 Fibonacci numbers.
"""

def calculate_fibonacci(n):
    """
    Calculate the first n Fibonacci numbers.

    Args:
        n (int): Number of Fibonacci numbers to calculate

    Returns:
        list: List of the first n Fibonacci numbers
    """
    if n <= 0:
        return []
    elif n == 1:
        return [0]

    fib_sequence = [0, 1]

    for i in range(2, n):
        next_fib = fib_sequence[i-1] + fib_sequence[i-2]
        fib_sequence.append(next_fib)

    return fib_sequence


if __name__ == "__main__":
    n = 20
    fibonacci_numbers = calculate_fibonacci(n)

    print(f"The first {n} Fibonacci numbers are:")
    for i, num in enumerate(fibonacci_numbers):
        print(f"F({i}) = {num}")
