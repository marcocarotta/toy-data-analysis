import numpy as np

def generate_data(N, sigma=0.1):
    """
    Generate N data points for polynomial curve fitting.
    
    Parameters:
    N (int): Number of data points to generate.
    sigma (float): Standard deviation of the noise.
    
    Returns:
    tuple: Arrays of input values (x) and target values (t).
    """
    # Generate N random points uniformly distributed in [0, 1]
    x = np.random.uniform(0, 1, N)
    # Generate N noise values from a normal distribution with mean 0 and std deviation sigma
    epsilon = np.random.normal(0, sigma, N)
    # Calculate the target values using the sine function and adding noise
    t = np.sin(2 * np.pi * x) + epsilon
    return x, t

# Example usage:
N = 50
x, t = generate_data(N)

def polynomial(x, w):
    """
    Calculate the polynomial function value for a given x and parameter vector w.
    
    Parameters:
    x (float): Input value.
    w (array): Parameter vector.
    
    Returns:
    float: Polynomial function value.
    """
    D = len(w) - 1  # Degree of the polynomial
    result = 0
    # Sum the polynomial terms up to degree D
    for k in range(D + 1):
        result += w[k] * (x ** k)
    return result

# Example usage:
D = 3  # Degree of the polynomial
w = np.random.randn(D + 1)  # Random initial parameter vector
y = [polynomial(xi, w) for xi in x]  # Calculate polynomial values for all x

def error_function(x, t, w):
    """
    Calculate the error function value for given input values, target values, and parameter vector.
    
    Parameters:
    x (array): Input values.
    t (array): Target values.
    w (array): Parameter vector.
    
    Returns:
    float: Error function value.
    """
    N = len(x)
    error = 0
    # Sum the squared differences between the polynomial and target values
    for i in range(N):
        error += 0.5 * (polynomial(x[i], w) - t[i]) ** 2
    return error

# Example usage:
E = error_function(x, t, w)

def gradient_error_function(x, t, w):
    """
    Calculate the gradient of the error function with respect to the parameter vector.
    
    Parameters:
    x (array): Input values.
    t (array): Target values.
    w (array): Parameter vector.
    
    Returns:
    array: Gradient vector.
    """
    N = len(x)
    D = len(w) - 1
    gradient = np.zeros(D + 1)
    # Sum the partial derivatives of the error function with respect to each parameter
    for i in range(N):
        error = polynomial(x[i], w) - t[i]
        for k in range(D + 1):
            gradient[k] += error * (x[i] ** k)
    return gradient

# Example usage:
grad_E = gradient_error_function(x, t, w)

def gradient_descent(x, t, w, learning_rate=0.01, iterations=1000):
    """
    Find the optimal parameter vector using gradient descent.
    
    Parameters:
    x (array): Input values.
    t (array): Target values.
    w (array): Initial parameter vector.
    learning_rate (float): Learning rate for gradient descent.
    iterations (int): Number of iterations for gradient descent.
    
    Returns:
    array: Optimized parameter vector.
    """
    for _ in range(iterations):
        # Calculate the gradient of the error function
        grad = gradient_error_function(x, t, w)
        # Update each parameter using the gradient
        for k in range(len(w)):
            w[k] -= learning_rate * grad[k]
    return w

# Example usage:
w_opt = gradient_descent(x, t, w)

def solve_linear_system(x, t, D):
    """
    Find the optimal parameter vector by solving the linear system of equations.
    
    Parameters:
    x (array): Input values.
    t (array): Target values.
    D (int): Degree of the polynomial.
    
    Returns:
    array: Optimized parameter vector.
    """
    N = len(x)
    A = np.zeros((D + 1, D + 1))
    T = np.zeros(D + 1)
    # Fill the matrix A and vector T with the appropriate sums
    for i in range(N):
        for k in range(D + 1):
            for j in range(D + 1):
                A[k, j] += x[i] ** (k + j)
            T[k] += x[i] ** k * t[i]
    # Solve the linear system to find the optimal parameters
    w_opt = np.linalg.solve(A, T)
    return w_opt

# Example usage:
w_opt_linear = solve_linear_system(x, t, D)

def plot_polynomial_fit(x, t, w, title):
    """
    Plot the data points and the polynomial fit.
    
    Parameters:
    x (array): Input values.
    t (array): Target values.
    w (array): Parameter vector.
    title (str): Title of the plot.
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(x, t, color='blue', label='Data points')
    x_range = np.linspace(0, 1, 100)
    y = [polynomial(xi, w) for xi in x_range]
    plt.plot(x_range, y, color='red', label='Polynomial fit')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('t')
    plt.legend()
    plt.show()

# Example usage:
# Gradient descent solution
w_opt = gradient_descent(x, t, w)
plot_polynomial_fit(x, t, w_opt, 'Polynomial Fit (Gradient Descent)')
# Linear system solution
w_opt_linear = solve_linear_system(x, t, D)
plot_polynomial_fit(x, t, w_opt_linear, 'Polynomial Fit (Linear System)')
    