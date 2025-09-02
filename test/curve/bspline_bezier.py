import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.special import comb

def bezier_curve(control_points, num_points=1000):
    """Generate a Bézier curve from control points.
    
    Args:
        control_points: Array of control points
        num_points: Number of points to generate on the curve
        
    Returns:
        Array of points on the Bézier curve
    """
    n = len(control_points) - 1  # Degree of the curve
    t = np.linspace(0, 1, num_points)
    curve_points = np.zeros((num_points, 2))
    
    for i in range(n + 1):
        # Bernstein polynomial
        bernstein = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        # Add contribution of each control point
        curve_points += np.outer(bernstein, control_points[i])
            
    return curve_points

def base_function(i, k, u, node_vector):
    """Calculate the basis function Bik(u) for B-spline curve.
    
    Args:
        i: Control point index
        k: Order of the B-spline (degree + 1)
        u: Parameter value
        node_vector: Knot vector
        
    Returns:
        Value of the basis function
    """
    if k == 1:  # 1st order, 0 degree B-spline
        if node_vector[i] <= u < node_vector[i+1] or (u == 1.0 and node_vector[i] <= u <= node_vector[i+1]):
            return 1.0
        else:
            return 0.0
    else:
        # Support interval lengths
        length1 = node_vector[i+k-1] - node_vector[i]
        length2 = node_vector[i+k] - node_vector[i+1]
        
        # Define 0/0 = 0
        if length1 == 0.0:
            length1 = 1.0
        if length2 == 0.0:
            length2 = 1.0
        
        # Recursive calculation
        term1 = 0.0
        if length1 != 0:
            term1 = (u - node_vector[i]) / length1 * base_function(i, k-1, u, node_vector)
            
        term2 = 0.0
        if length2 != 0:
            term2 = (node_vector[i+k] - u) / length2 * base_function(i+1, k-1, u, node_vector)
            
        return term1 + term2

def bspline_gen(control_points, n, k, node_vector, accuracy, weights):
    """Generate points on a B-spline curve.
    
    Args:
        control_points: List of control points
        n: Number of control points
        k: Order of the B-spline
        node_vector: Knot vector
        accuracy: Step size for parameter u
        weights: Weights for control points
        
    Returns:
        Array of points on the B-spline curve
    """
    num = int(1 / accuracy)
    curve_points = np.zeros((num+1, 2))
    
    for idx, u in enumerate(np.linspace(0, 1, num+1)):
        bik_w = np.zeros(n)
        dom = 0
        
        for i in range(n):
            bik_w[i] = base_function(i, k, u, node_vector) * weights[i]
            dom += bik_w[i]
            
        if dom != 0:
            for i in range(n):
                curve_points[idx] += (bik_w[i] / dom) * control_points[i]
    
    return curve_points

def bspline(control_points, k, line_type='b-', line_width=2):
    """Create and plot a B-spline curve.
    
    Args:
        control_points: Control points array
        k: Order of the B-spline
        line_type: Line style for plotting
        line_width: Line width for plotting
        
    Returns:
        Array of points on the B-spline curve
    """
    n = len(control_points)
    weights = np.ones(n) / n
    # Create uniform B-spline knot vector
    node_vector = np.linspace(0, 1, n+k)
    
    # Generate the B-spline curve
    curve_points = bspline_gen(control_points, n, k, node_vector, 0.001, weights)
    
    # Plot the curve (if plt is available)
    if 'matplotlib.pyplot' in sys.modules:
        plt.plot(curve_points[:, 0], curve_points[:, 1], line_type, linewidth=line_width)
    
    return curve_points

# Main code
# Control point sequence P
P = np.array([
    [0, 0],
    [4, 6],
    [6, 2],
    [12, 3],
    [15, 5],
    [22, 9],
    [27, 2],
    [33, 6],
    [38, 3],
    [42, 0],
    [45, 4]
])

# P2 is P with one modified point
P2 = P.copy()
P2[5, 1] = 5  # Modify the 6th point's y-coordinate from 9 to 5

# Create figure with two subplots for comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# First subplot: B-spline comparison
ax1.plot(P[:, 0], P[:, 1], 'k--o', label='Control Points')
# Plot both B-spline curves
bspline_curve1 = bspline(P, 3, 'b-', 2)
bspline_curve2 = bspline(P2, 3, 'r-', 2)

ax1.legend(['Control Points', 'Original B-Spline', 'Modified B-Spline'])
ax1.set_title('B-Spline Curve - Local Influence of Control Points')
ax1.grid(True)
ax1.set_aspect('equal')

# Second subplot: Bezier curve comparison
ax2.plot(P[:, 0], P[:, 1], 'k--o', label='Control Points')
ax2.plot(P2[:, 0], P2[:, 1], 'k--x', label='Modified Control Points')

# Plot both Bezier curves
bezier_points1 = bezier_curve(P)
bezier_points2 = bezier_curve(P2)
ax2.plot(bezier_points1[:, 0], bezier_points1[:, 1], 'b-', linewidth=2, label='Original Bezier')
ax2.plot(bezier_points2[:, 0], bezier_points2[:, 1], 'r-', linewidth=2, label='Modified Bezier')

ax2.legend(['Control Points', 'Modified Control Points', 'Original Bezier', 'Modified Bezier'])
ax2.set_title('Bezier Curve - Global Influence of Control Points')
ax2.grid(True)
ax2.set_aspect('equal')

plt.tight_layout()
plt.show()

# Additional visualization showing the difference more clearly
plt.figure(figsize=(12, 6))
# Plot control points
plt.plot(P[:, 0], P[:, 1], 'k--o', label='Control Points')
plt.plot(P2[5, 0], P2[5, 1], 'kx', markersize=10, label='Modified Point')

# Plot curves
plt.plot(bspline_curve1[:, 0], bspline_curve1[:, 1], 'b-', linewidth=2, label='Original B-Spline')
plt.plot(bspline_curve2[:, 0], bspline_curve2[:, 1], 'r-', linewidth=2, label='Modified B-Spline')
plt.plot(bezier_points1[:, 0], bezier_points1[:, 1], 'g--', linewidth=2, label='Original Bezier')
plt.plot(bezier_points2[:, 0], bezier_points2[:, 1], 'm--', linewidth=2, label='Modified Bezier')

# Highlight the affected areas
plt.title('Comparison of B-Spline (Local Influence) vs Bezier (Global Influence)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()