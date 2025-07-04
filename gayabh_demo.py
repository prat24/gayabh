# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import basinhopping

#test non-convex function
def objective_func(x):
    return x[0]**2 + x[1]**2 + 0.5*np.sin(4*x[0]) + 0.5*np.sin(4*x[1])

#computing gradient by hand
def grad_func(x):
    #derivative of x^2 + 0.5*sin(4*x) with respect to x
    grad_x = 2*x[0] + 0.5*4*np.cos(4*x[0])
    grad_y = 2*x[1] + 0.5*4*np.cos(4*x[1])
    return np.array([grad_x, grad_y])

class GayabhStepper:
    """
    Gradient Ascent Yield Assisted Basin Hopping
    direct hops towards a mix between gradient direction and a random vector direction based on weight
    """

    def __init__(self, step_size, grad_w):
        self.step_size = step_size
        self.grad_w = grad_w
        self.rand_w = 1.0 - grad_w #remaining weight

    def __call__(self, curr_x):
        """called by basinhopping to get next step"""
        grad = grad_func(curr_x) #get gradient
        print(f"gradient at {curr_x}: {grad}")

        #get unit gradient
        grad_mag = np.sqrt(grad[0]**2 + grad[1]**2)
        if grad_mag > 0.0001:
            grad_unit = grad / grad_mag
        else: #if gradient is 0 (can't use) then take random direction
            grad_unit = np.random.randn(2)
            grad_unit = grad_unit / np.linalg.norm(grad_unit)

        #make unit random vector
        rand_vec = np.random.randn(2)
        rand_unit = rand_vec / np.linalg.norm(rand_vec)

        #combine to unit direction vector
        mix_dir = (self.grad_w * grad_unit +
                          self.rand_w * rand_unit)
        mix_unit = mix_dir / np.linalg.norm(mix_dir)

        #take step
        new_x = curr_x + self.step_size * mix_unit

        print(f"  stepping from {curr_x} to {new_x}")
        return new_x

def test_gayabh():
    #setup
    start = np.array([3.0, 2.0])
    num_iter = 18
    step_size = 1.0
    grad_w = 0.73  #weight hyperparameter

    print(f"starting at: {start}")
    print(f"starting value: {objective_func(start):.4f}")
    print(f"gradient weight: {grad_w}")

    stepper = GayabhStepper(step_size, grad_w) #initialize

    np.random.seed(13)

    result = basinhopping(
        objective_func,
        start,
        niter=num_iter,
        T=1.0,
        take_step=stepper #use our custom stepper
    )

    print("optimization finished!")
    print(f"final point: [{result.x[0]:.4f}, {result.x[1]:.4f}]")
    print(f"final value: {result.fun:.6f}")
    print(f"total function calls: {result.nfev}")

    #check
    if result.fun < 0.1:
        print("found good min!")
    else:
        print("try something else bro")

def onestep_demo():
    print("\n one gayabh:")

    curr_step = np.array([1.0, 1.5])
    stepper = GayabhStepper(step_size=0.5, grad_w=0.7)

    print(f"current point: {curr_step}")
    print(f"current function value: {objective_func(curr_step):.4f}")

    grad = grad_func(curr_step)
    print(f"gradient: {grad}")
    print(f"gradient magnitude: {np.linalg.norm(grad):.4f}")

    np.random.seed(13)
    next = stepper(curr_step)

    print(f"next point: {next}")
    print(f"next function value: {objective_func(next):.4f}")

    step_vec = next - curr_step
    print(f"step taken: {step_vec}")

if __name__ == "__main__":
    test_gayabh()
    onestep_demo()