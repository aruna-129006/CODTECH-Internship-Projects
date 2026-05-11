from pulp import *

# Create optimization problem
problem = LpProblem("Factory_Optimization", LpMaximize)

# Decision variables
x = LpVariable("Product_A", lowBound=0, cat='Integer')
y = LpVariable("Product_B", lowBound=0, cat='Integer')

# Objective function
problem += 40 * x + 30 * y, "Profit"

# Constraints
problem += 2 * x + 1 * y <= 100   # Labor hours
problem += 1 * x + 1 * y <= 80    # Machine hours

# Solve problem
problem.solve()

# Print results
print("Status:", LpStatus[problem.status])

print("Optimal Production:")
print("Product A =", x.varValue)
print("Product B =", y.varValue)

print("Maximum Profit = ", value(problem.objective))
