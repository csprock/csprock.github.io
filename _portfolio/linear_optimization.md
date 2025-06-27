---
title: "Introduction to Linear Optimization in Python"
excerpt: "An Introduction"
header:
  teaser: /assets/images/linear_optimization.png
toc: false
toc_label: "Sections"
toc_sticky: true
classes:
    - wide
math: true
---


# Introduction to Linear Optimization in Python

Linear optimization (also known as linear programming) is a powerful mathematical technique used to find the best outcome in a mathematical model whose requirements are represented by linear relationships. It's a cornerstone of operations research that helps organizations make optimal decisions when faced with complex constraints and competing objectives.

As a data scientist or analyst, you've likely encountered problems where you need to maximize or minimize an outcome subject to various constraints. Whether it's optimizing a supply chain, allocating resources, or planning production schedules, linear optimization provides a structured approach to finding optimal solutions.

In this notebook, we'll explore how to formulate and solve linear optimization problems using Pyomo, an open-source Python-based optimization modeling language. Pyomo offers a flexible and intuitive way to express optimization problems and connects seamlessly with various solvers. 

The example that we will use to motivate our exploration comes from the petroleum industry. There are hundreds of different grades of crude oil in the world, each with different chemical properties. Oil refineries are configured to run using grades whose chemical properties fall within certain bounds. Often, raw crudes from different sources are not suitable for direct use in the refinery, but by blending different grades in the right ratios, a crude oil blend with suitable chemical properties can be created for use in the refinery. 

#### What You Will Learn

* The essential parts of a linear optimization model
* How to translate the problem statement into math and into code
* How to use basic algebra to convert a non-linear problem into a linear one
* How to program variables and constraints that are integers
* How to parse the solution results



```python
%cd /app

import json
import pyomo.environ as pyo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.blend_indices import blend_indices

solver = pyo.SolverFactory('glpk')

CONSTRAINTS = dict()
```

    /app


# Model Formulation 

Linear optimization problems have five key components: *parameters*, *variables*, *constraints*, *sets* and the *objective function*.

1. **Parameters** define fixed data in the model
2. **Variables** are the quantities to be optimized
3. **Constraints** define the space of possible solutions
4. **Sets** are used to index parameters, variables and constraints
5. **Objective Function** defines the quantity to be minimized or maximized


## Sets

Sets are used to index over parameters, variables and constraints. 

1. *grades* of crude (denoted $C$) and 
2. chemical *qualities* (denoted $Q$) which are different chemical properties of a grade of crude oil


```python
with open("data/index_constraints.json", "r") as f:
    CONSTRAINTS['quality'] = json.load(f)

print("Quality Constraints:")
for k, v in CONSTRAINTS['quality'].items():
    print(f"{k}: {[round(x,4) for x in v]}")

with open("data/index_assay.json", "r") as f:
    index_assay = json.load(f)
```

    Quality Constraints:
    ibp220: [0.0, 19.8]
    htsd50: [470, 570]
    1020plus: [0, 16]
    v: [0, 15]
    ni: [0.0, 8.0]
    neut: [0.0, 0.28]
    mcrt: [0.0, 2.4]
    vis100f_idx: [-2.3644, 0.2514]
    rvp_idx: [0.0, 16.6784]
    sg: [0.8156, 0.8398]
    sulf: [0.0, 0.4185]
    pour_idx: [6.521, 100.0]



```python
## Define Sets ###
model = pyo.ConcreteModel()

model.CRUDES = pyo.Set(initialize=[c for c in index_assay.keys()])
model.QUALITIES = pyo.Set(initialize=[q for q in CONSTRAINTS['quality'].keys()])

model.CRUDES.pprint()
model.QUALITIES.pprint()
```

    CRUDES : Size=1, Index=None, Ordered=Insertion
        Key  : Dimen : Domain : Size : Members
        None :     1 :    Any :   10 : {'crude_1', 'crude_2', 'crude_3', 'crude_4', 'crude_5', 'crude_6', 'crude_7', 'crude_8', 'crude_9', 'crude_10'}
    QUALITIES : Size=1, Index=None, Ordered=Insertion
        Key  : Dimen : Domain : Size : Members
        None :     1 :    Any :   12 : {'ibp220', 'htsd50', '1020plus', 'v', 'ni', 'neut', 'mcrt', 'vis100f_idx', 'rvp_idx', 'sg', 'sulf', 'pour_idx'}


## Variables and Parameters

### Parameters

The blend model has two sets of parameters, the qualities of individual crude grades and the prices of each grade. For example, each grade will have its own API gravity, sulfer content and viscosity. Each grade will also have its own price. Denote

1. the value of quality $j \in Q$ for crude $i \in C$ as $q_{ij}$
2. the price of crude $i \in C$ as $p_i$

The prices and qualities are fixed quantities of the model and are initialized from data. 


```python
# set grade prices to values between 65 and 75
np.random.seed(1234)

GRADE_PRICES = {c: np.random.uniform(low=65,high=75) for c in model.CRUDES}

print("Grade Prices:")
for k, v in GRADE_PRICES.items():
    print(f"{k}: ${round(v,2)}")
```

    Grade Prices:
    crude_1: $66.92
    crude_2: $71.22
    crude_3: $69.38
    crude_4: $72.85
    crude_5: $72.8
    crude_6: $67.73
    crude_7: $67.76
    crude_8: $73.02
    crude_9: $74.58
    crude_10: $73.76



```python
### Define Parameters ###
def initialize_coefficients(model, crude, quality):
    return float(index_assay[crude][quality])

model.qualities = pyo.Param(model.CRUDES * model.QUALITIES, initialize=initialize_coefficients, domain=pyo.Reals)

def initialize_prices(model, crude):
    return GRADE_PRICES[crude]

model.prices = pyo.Param(model.CRUDES, initialize=initialize_prices, domain=pyo.Reals)
```

### Variables

Let $x_i$ denote the *quantity* of crude $i \in C$ (measured in barrels). These variables are what the model solves for. We also place constraints on the values each variable can take based on the type of problem we are solving. Since we cannot purchas a negative amount of crude, we restrict $x_i$ to be non-negative

$$
x_i \geq 0
$$

We also allow the user to set a custom supply of each crude, denoted $U_i$. We can optionally also force the model to use a minimum amount of crude, denoted $L_i$. Together, these constraints combine to 

$$
0 \leq L_i \leq x_i \leq U_i \ \ \ \text{for all} \ \ i \in C
$$


```python
CONSTRAINTS['supply'] = {
    'crude_1': [0, 300000],
    'crude_2': [0, 300000],
    'crude_3': [0, 300000],
    'crude_4': [0, 300000],
    'crude_5': [0, 300000],
    'crude_6': [0, 300000],
    'crude_7': [0, 300000],
    'crude_8': [0, 300000],
    'crude_9': [0, 300000],
    'crude_10': [0, 300000]
}
```

Pyomo used an object model to represent variables. The variable takes an iterable (in this case, we use the CRUDES set of the model, which was initialized earlier), a domain (again, using Pyomo's object model) and a function that initializes the bounds. Under the hood, Pyomo will iterate over the set of indices, calling the inialization function, which must accept the model object and an index and return a tuple for the constraint values. 


```python
def initialize_bounds(model, i):
    return (CONSTRAINTS['supply'][i][0], CONSTRAINTS['supply'][i][1])


model.x = pyo.Var(model.CRUDES, domain=pyo.NonNegativeReals, bounds=initialize_bounds)
```

## Constraints

Constraints are imposed on the set of possible solutions in order to meet business criteria. The first step in creating the constraints is to state the business objects in words and then translate those into mathematical formulas. Our blend model must meet the following business objectives: 

1. the total amount of blended crude must be no greater than the tank capacity
2. we want to know the per barrel profit *and loss* from blending
3. the blended grade must meet the required quality specifications
4. in practice there must never be more than three grades in a single blend

### 1. Tank Capacity Constraint

The amount of blended crude cannot exceed the tank capacity $T$. Since the total volume of the blend is simply the sum of the component volumes, this constraint can be expressed as 

$$
\sum_{i \in C} x_i \leq T
$$

### 2. Computing the Profit (or Loss) from Blending

We are interested in the profit on the optimal blend, *even if the profit is negative (a loss)*. This is important for seeing how close conditions are to making blending profitable. Recall that we imposed the constraint that $x_i \geq 0$. If there is no profitable blend, the model will set $x_i = 0$ for all $i$ unless we impose the constraint that the model is required to blend a non-zero amount of crude. The constaint amounts to setting a lower bound on the total blended volume. The number can be an arbitrary positive number but in this example we set it to be 1 because it will make calculation of the the per-barrel loss easy when there is exactly one barrel in the blend. 

$$
\sum_{i \in C} x_i > 1
$$

The combined constraint is

Next we will set our first constraint. This will demonstrate the power of Pyomo's object model. We can express the constraint in code the way we would in mathematical notation, using the `sum` function over the model variables. Pyomo will store this expression and convert it into input accepted by one of the underlying solvers. We also include the upper and lower bounds with the sum in the middle, similarly to how we would express an inequality in mathematical notation. 

The Pyomo `Constraint` object accepts a function that accepts the model object and returns the constraint. 


```python
## total constraint 
CONSTRAINTS['total'] = 500000

def total_barrel_constraint(model):
    min_ = 10.0
    return (min_, sum(model.x[i] for i in model.CRUDES), CONSTRAINTS['total'])

model.total_barrel_constraint = pyo.Constraint(rule=total_barrel_constraint)
```

### 3. Quality Constraints

We assume that all qualities blend linearly, or that they have been converted to blend indices which do. A quality blends linearly if it can expressed as a weighted average. Let $L_j$ and $U_j$ be the upper and lower bounds on the $j$-th quality constraint and let $q_{ij}$ be the $j$-th quality of the $i$-th crude.

#### Volume-Based Qualities
Most of the quality constraints are *volume-based* so that the weights iin the weighted average are expressed as a fraction of the total volume in the blend (measured in barrels). 

$$
L_j \leq \sum_{i \in C} \frac{q_{ij} x_i}{\sum_{i\in C} x_i} \leq U_j
$$


#### Weight-Based Qualities

Some of the qualities are blended on a *weight-basis* rather than a volume basis. Therefore, the volume of the crude needs to be converted to a weight by multiplying by the *specific gravity* of the grade. 

$$
L_j \leq \sum_{i \in C} \frac{q_{ij} (s_i x_i)}{\sum_{i\in C} s_i x_i} \leq U_j
$$

#### Linearizing The Ratio Constraints

The numerators and denominators of the above constraints are both linear, however their ratio is not. We can use a non-linear solver to solve the problem with the current constraints, however nonlinear problems are more complex and subject to issues such as local optimiums and numerical instability. The alternative is to reformulate these constraints to make them linear. 

For example, to linearize the upper bound of the a volume-based constraint, we can use some simple algebra to rearrange the terms

$$
\begin{align*}

\sum_{i \in C} \frac{q_{ij}x_i}{\sum_{i \in C}x_i} &\leq U_j \\ 
\sum_{i \in C} q_{ij}x_i &\leq U_j \sum_{x\in C} x_i \\
\sum_{i \in C} q_{ij} x_i - U_j \sum_{x \in C} x_i &\leq 0 \\
\sum_{i \in C} (q_{ij} - U_j)x_i &\leq 0
\end{align*}
$$

which is linear. Similarly, the lower bounds can be expressed as 

$$
\sum_{i \in C} (q_{ij} - L_j)x_i \geq 0
$$

Below we write express the above constraints in code. Constraints that have no upper or lower bound can use the familiar Python inequality operators to express the equations intuitively in code. 


```python
_volume_constraints = ['rvp_idx', 'sg', 'pour_idx']
def quality_constraint_upper_bound(model, q):
    if q in _volume_constraints:
        return sum(model.x[c] * (model.qualities[c, q] - CONSTRAINTS['quality'][q][1]) for c in model.CRUDES) <= 0
    else:
        return sum(model.qualities[c, "sg"] * model.x[c] * (model.qualities[c, q] - CONSTRAINTS['quality'][q][1]) for c in model.CRUDES) <= 0
    
def quality_constraint_lower_bound(model, q):
    if q in _volume_constraints:
        return sum(model.x[c] * (model.qualities[c, q] - CONSTRAINTS['quality'][q][0]) for c in model.CRUDES) >= 0
    else:
        return sum(model.qualities[c, "sg"] * model.x[c] * (model.qualities[c, q] - CONSTRAINTS['quality'][q][0]) for c in model.CRUDES) >= 0
    
model.quality_constraint_ub = pyo.Constraint(model.QUALITIES, rule=quality_constraint_upper_bound)
model.quality_constraint_lb = pyo.Constraint(model.QUALITIES, rule=quality_constraint_lower_bound)
```

### 4. Constraints on the Number of Blend Components

In practice, there are rarely more than three components in a blend. In order to formulate a constraint that accomplishes this, we introduce binary indicator variables for each crude $\delta_i \in \{0, 1\}$ that indicate if that is included in the blend. 

$$
\delta_i = 
\begin{cases} 
1 & \text{if crude } i \text{ is included in the blend } (x_i > 0) \\
0 & \text{otherwise } (x_i = 0)
\end{cases}
$$

The constraint on the number of blend components can be expressed as:

$$
\sum_{i \in C} \delta_i \leq M
$$

where $M$ is the maximum number of crudes allowed in the blend (in our case, $M=3$).



```python
model.xi = pyo.Var(model.CRUDES, domain=pyo.Binary)

def max_crudes_constraint(model):
    return sum(model.xi[c] for c in model.CRUDES) <= 3

model.max_crudes_constraint = pyo.Constraint(rule=max_crudes_constraint)
```

In order to connect these binary variables to the volume variables $x_i$, we must introduce constraints that force $x_i = 0$ when $\delta_i =0$. To do this, we introduce the follwing constraints

$$
x_i - \delta_i U_i \leq 0
$$

To see how these constraints work, recall that $U_i > 0$ is the upper limit on the available supply of crude $i$. Also recall that $x_i \geq 0$ since we cannot have negative volumes. Together these imply that if $\delta_i = 0$, then we must have 

$$
\begin{align*}

x_i - (0)U_i \leq 0 &\Longrightarrow \\ x_i \leq 0 &\Longrightarrow x_i = 0

\end{align*}
$$

Conversely, if $x_i = 0$, then we must have 

$$
\begin{align*}

(0) - \delta_i U_i \leq 0 &\Longrightarrow \\ -\delta_i U_i \leq 0 &\Longrightarrow \\ \delta_i = 0 

\end{align*}
$$

Therefore, $\delta_i = 0$ if and only if $x_i = 0$. 


```python
def binary_constraint(model, c):
    return model.x[c] - CONSTRAINTS['supply'][c][1] * model.xi[c] <= 0

model.binary_constraint = pyo.Constraint(model.CRUDES, rule=binary_constraint)
```

## The Objective Function

Our objective is to maximize profit, which is the difference between the price of a barrel of the blended crude and the cost of acquiring the individual blend grades.

$$
\max_{x_i} \Big\{ \sum_{i \in C} (p_{blend} - p_i) x_i \Big\}
$$


```python
BLEND_GRADE_PRICE = 80
```


```python
def profit_maximization(model):
    return sum((BLEND_GRADE_PRICE - model.prices[c]) * model.x[c] for c in model.CRUDES)

model.profit = pyo.Objective(rule=profit_maximization, sense=pyo.maximize)
```

## Solving the Model


```python
# some helper functions

def get_selected_grades(model_instance):
    """
    Return the subset of grades selected by the model.
    """
    return [c for c in model_instance.CRUDES if model_instance.xi[c].value == 1]


def get_amounts(model_instance):
    """
    Return the amount of each grade selected by the model.
    """

    amounts = {}
    total = 0
    for c in model_instance.CRUDES:
        v = pyo.value(model_instance.x[c])
        amounts[c] = v
        total += v

    amounts['total'] = total
    return amounts


def get_profit(model_instance):
    """
    Return the profit of the selected grades.
    """
    amounts = get_amounts(model_instance)
    total = amounts['total']
    profit = pyo.value(model_instance.profit)
    profit_per_barrel = profit / total
    return {'profit': profit, 'profit_per_barrel': profit_per_barrel}


def get_ratios(model_instance):
    """
    Return the ratios of the selected grades.
    """
    amounts = get_amounts(model_instance)
    total = amounts['total']
    ratios = {c: v / total for c, v in amounts.items() if c != 'total'}
    return ratios

def get_blend_quality(model_instance):

    constraint_values = dict()

    total_volume = sum(pyo.value(model_instance.x[c]) for c in model_instance.CRUDES)
    total_weight = sum(pyo.value(model_instance.qualities[c, 'sg']) * pyo.value(model_instance.x[c]) for c in model_instance.CRUDES)

    for q in model_instance.QUALITIES:
        # volume-based quantities
        if q in ['rvp_idx', 'sg', 'pour_idx']:
            num = sum(pyo.value(model_instance.x[c]) * (pyo.value(model_instance.qualities[c, q])) for c in model_instance.CRUDES)
            value = num / total_volume
        else:
            num = sum(pyo.value(model_instance.qualities[c, 'sg']) * pyo.value(model_instance.x[c]) * pyo.value(model_instance.qualities[c, q]) for c in model_instance.CRUDES)
            value = num / total_weight
        
        lower_bound = CONSTRAINTS['quality'][q][0]
        upper_bound = CONSTRAINTS['quality'][q][1]


        constraint_values[q] = {
            'blend_value': value,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }

    return constraint_values
```


```python
# create a model instance and pass it to the solver
model_instance = model.create_instance()
results = solver.solve(model_instance, tee=False)
```


```python
selected_grades = get_selected_grades(model_instance)
selected_grades
```




    ['crude_1', 'crude_3', 'crude_7']




```python
amounts = get_amounts(model_instance)
ratios = get_ratios(model_instance)

for c in selected_grades:
    # format ratios at percentages
    print(f"{c}: {round(amounts[c],2)} barrels - {round(100*ratios[c],2)}%")
```

    crude_1: 300000.0 barrels - 60.0%
    crude_3: 6186.36 barrels - 1.24%
    crude_7: 193813.64 barrels - 38.76%



```python
profit_amounts = get_profit(model_instance)

# Format profit amounts as dollar amounts with commas
total_profit = f"${profit_amounts['profit']:,.2f}"
profit_per_barrel = f"${profit_amounts['profit_per_barrel']:.2f}"

print(f"Total profit: {total_profit}")
print(f"Profit per barrel: {profit_per_barrel}")
```

    Total profit: $6,362,536.80
    Profit per barrel: $12.73



```python
blend_qualities = get_blend_quality(model_instance)

print("Blend Qualities:")
for k, v in blend_qualities.items():
    print(f"{k}: [{v['lower_bound']}, {round(v['blend_value'],4)}, {v['upper_bound']}]")
```

    Blend Qualities:
    ibp220: [0.0, 15.6672, 19.8]
    htsd50: [470, 541.2938, 570]
    1020plus: [0, 4.388, 16]
    v: [0, 9.5128, 15]
    ni: [0.0, 4.0035, 8.0]
    neut: [0.0, 0.1571, 0.28]
    mcrt: [0.0, 0.9491, 2.4]
    vis100f_idx: [-2.3643781672184256, 0.1342, 0.2513713768778552]
    rvp_idx: [0.0, 9.1366, 16.678404656440467]
    sg: [0.8155619596541787, 0.8156, 0.8397626112759644]
    sulf: [0.0, 0.4037, 0.4185]
    pour_idx: [6.520991515006882, 51.7075, 99.99999999999994]


## References 

If you want to check out the docker image that contains all the optimizers, see this [Dockerfile](https://github.com/csprock/dockerfiles/blob/master/Dockerfile-python-opt). 
