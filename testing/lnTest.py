import math


def ln(x):
    # Check that x is positive
    if x <= 0:
        raise ValueError("x must be positive")

    # Initialize the sum to 0 and the counter to 1
    result = 0
    n = 1

    # Use the identity ln(x) = ln(x/e) + ln(e) to reduce x to a value between 0 and 1
    while x > 1:
        # Apply the identity
        x = x/2.718
        result += 1

    # Use the Taylor series expansion to approximate ln(x)
    while True:
        # Calculate the next term in the series and add it to the result
        next_term = ((-1)**(n+1)) * (x-1)**n / n
        result += next_term

        # If the next term is smaller than the tolerance, return the result
        if abs(next_term) < 1e-6:
            return result

        # Increment the counter
        n += 1


result = ln(50)
print(result)
