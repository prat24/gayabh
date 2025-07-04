# Gradient Ascent Yield Assisted Basin Hopping (GAYABH)

## Global Optimization
Gradient descent, first proposed by Cauchy in 1847, has been the workhorse of optimization.  In modern machine learning, stochastic variants of gradient descent (SGD) remain ubiquitous.  However, basic gradient-based methods are inherently *local* – they follow the steepest downhill path and converge to the nearest valley.  In a complex, non-convex landscape, this means “you’re gonna converge to the local minimum, and there’s no way gradient descent will get you out of there”.  In practice one uses tricks like random restarts, momentum or adaptive steps (Adam, RMSProp, etc.) to escape shallow traps, but pure gradient descent was never meant to *guarantee* a global optimum in a multimodal loss.

By contrast, **global optimization** seeks the absolute best solution across *all* basins.  Early global methods like simulated annealing and genetic algorithms make random jumps to explore the space.  A particularly influential strategy is **basin-hopping**.  Basin-hopping was introduced in 1997 by Wales & Doye and used heavily in computational chemistry to find lowest-energy molecular structures.  Its recipe is simple: **(1)** apply a local minimizer from the current point to settle into a basin, **(2)** take a random “hop” (perturbation) to move to a new point, **(3)** accept or reject this new basin based on a Metropolis criterion, and repeat.  Intuitively, basin-hopping “transforms the complex energy landscape into a collection of basins” and uses random jumps to explore between them.  This makes it effective for problems where there are many valleys separated by tall barriers – it is “extremely efficient for … problems in physics and chemistry” with many minima.

In practice, however, basin-hopping is *not* widely used for high-dimensional ML models.  Its main drawback is scalability: as dimension grows, the number of local minima explodes exponentially.  Each random jump-and-minimize step is costly, and the algorithm requires many samples to confidently find the global best.  So while basin-hopping is conceptually appealing, it can become prohibitively expensive on large problems.

## Gradient Ascent Yield Assisted Basin Hopping

To bridge the gap between local gradient methods and global basin-hopping, we designed GAYABH.  The idea is to **bias the random hops towards the direction of steepest ascent** (the negative gradient direction) instead of purely random jumps. 

In effect, we “blend” a gradient-based move with a random jump. More formally this can be described as:

1. Compute the gradient **g = ∇f(x)** of the objective (note we treat this as an ascent direction).
2. If ∥g∥ is non-negligible, set **a = g/∥g∥** (normalized gradient direction); otherwise pick a random unit vector **a**.
3. Pick another independent random unit vector **r**.
4. Form the hop direction **h = normalize(w·a + (1–w)·r)**, where *w*∈\[0,1] is a weight (e.g. 0.8).
5. Step to the new point **x′ = x + α·h** (with hop size α).
6. Perform a local minimization from x′ (as in standard basin-hopping).
7. Accept or reject x′ using the Metropolis test (higher-energy moves are allowed with small probability).

In pseudocode:

```
g = gradient(x)                 # ∇f at current point
if norm(g) > tol:
    a = g / norm(g)            # ascent direction
else:
    a = random_unit_vector()
r = random_unit_vector()
h = normalize(w*a + (1-w)*r)   # blended direction
x_new = x + alpha * h
x_min = local_minimize(f, x_new)  # settle into new basin
if accept(x_min): 
    x = x_min
```

The key difference from classical basin-hopping is simply in how the hop is performed: instead of hopping in a completely random direction, we **partially guide the jump along the gradient**. The parameter *w* (0≤w≤1) controls the mix between gradient ascent and random exploration.  When w=1, the hop is purely gradient-driven; when w=0 it reduces to random hopping. The weight may be tuned.

## Why?

The intuition is that **large peaks or ridges** often act as barriers in the search space.  A purely local descent (gradient) will get stuck in one basin, and a purely random hop may frequently waste moves in uninformative directions.  By climbing first (gradient ascent) and then jumping, we effectively start each jump from a higher elevation.  Imagine trying to escape a valley surrounded by tall hills: if you climb partway up a hill (using the gradient) before jumping, you have a better chance of leaping over the ridge into a new basin.  In other words, aligning some of the random hops toward the steepest climb can “unlock” access to distant regions that would be very unlikely under purely random hops.  In our chemical-physics toy problem (a 2D “hilly” function), this meant the optimizer tended to hop toward promising peaks and then slide down into deeper minima, often finding the global minimum more quickly in low dimensions.

Mathematically, the **hop direction** $\hat{\mathbf{h}}$ is defined as:  

$$
\hat{\mathbf{h}} = \frac{w \cdot \hat{\mathbf{g}} + (1 - w) \cdot \hat{\mathbf{u}}}{\left\| w \cdot \hat{\mathbf{g}} + (1 - w) \cdot \hat{\mathbf{u}} \right\|}
$$

With:  
- Normalized gradient direction:
  $$
  \hat{\mathbf{g}} =
  \begin{cases}
  \dfrac{\nabla f(\mathbf{x})}{\|\nabla f(\mathbf{x})\|}, & \text{if } \|\nabla f(\mathbf{x})\| > \epsilon \\[1.2em]
  \dfrac{\mathbf{u}_1}{\|\mathbf{u}_1\|}, & \text{otherwise}
  \end{cases}
  $$

- Random unit vector:
  $$
  \hat{\mathbf{u}} = \frac{\mathbf{u}_2}{\|\mathbf{u}_2\|}, \quad \mathbf{u}_2 \sim \mathcal{U}(-1,1)^n
  $$

We then move $x \leftarrow x + \alpha h$, followed by a local minimization.  By construction, when the gradient is strong, $h$ is close to $\nabla f$; when the gradient is weak or at a flat point, $h$ defaults to a near-random direction.  This strategy can be seen as a **hybrid** global-local method, akin to proposals of combining basin-hopping with gradient steps, but here we incorporate the gradient into the hop itself.

## Performance

Empirically, we observed that this gradient-biased hopping worked well in **low-dimensional** tests (e.g. the 2D example above). However, in **high dimensions** the advantage faded.  Speculatively, there are two culprits:

* **Explosion of basins:** As dimension grows, the number of local minima explodes (often exponentially).  The landscape becomes so rich that any guided jump struggles to explore adequately.  Basin-hopping itself faces a severe scalability issue: the computational cost and memory needed grow very quickly with dimensionality.  Our gradient-biased method is not exempt from this curse.

* **Geometry of high-D space:** In high dimensions, a random vector is almost orthogonal to any fixed direction on average.  In practice this means our combined direction $h = \mathrm{normalize}(w\,a + (1-w)\,r)$ is often dominated by the random part unless $w$ is 1.  The gradient direction $a$ contributes less to the overall hop direction when $r$ is nearly perpendicular.  Thus, the “gradient-assisted” bias becomes weaker in higher dimensions, and the algorithm behaves more like standard basin-hopping.
