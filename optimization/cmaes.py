import traceback
import torch
from typing import Callable, Tuple

from cantrips.logging.logger import get_logger

torch.set_printoptions(precision=3, sci_mode=False)

logger = get_logger()


class CMAES:
    def __init__(
        self,
        dimension: int,
        population_size: int = None,
        initial_mean: torch.Tensor = None,
        initial_sigma: float = 1.0,
        device: str = "cuda:0",
    ):
        """
        Initialize CMA-ES optimizer

        Args:
            dimension: Number of parameters to optimize
            population_size: Number of samples per generation (default: 4 + floor(3 * log(dimension)))
            initial_mean: Initial mean vector (default: zeros)
            initial_sigma: Initial step size (default: 1.0)
            device: Device to run computations on ('cpu' or 'cuda')
        """
        self.dimension = dimension
        self.population_size = population_size or 4 + int(
            3 * torch.log(torch.tensor(dimension)).item()
        )
        self.device = device

        # Strategy parameters
        self.mean = (
            initial_mean
            if initial_mean is not None
            else torch.zeros(dimension, device=self.device)
        )
        self.sigma = initial_sigma
        self.covariance = torch.eye(
            dimension, device=self.device
        )  # Initialize covariance matrix as identity

        # Evolution path parameters
        self.pc = torch.zeros(dimension, device=self.device)
        self.ps = torch.zeros(dimension, device=self.device)

        # Learning rates and constants
        self.cc = 4 / (dimension + 4)
        self.cs = 4 / (dimension + 4)
        self.c1 = 2 / ((dimension + 1.3) ** 2)
        self.cmu = min(
            1 - self.c1,
            2
            * (self.population_size - 2 + 1 / self.population_size)
            / ((dimension + 2) ** 2),
        )
        self.damps = (
            1
            + 2
            * max(
                0,
                torch.sqrt(torch.tensor((self.population_size - 1) / (dimension + 1)))
                - 1,
            )
            + self.cs
        )

        # Weights for rank-based update
        weights = torch.log(torch.tensor(self.population_size + 0.5)) - torch.log(
            torch.arange(1, self.population_size + 1, device=self.device)
        )
        self.weights = weights / weights.sum()
        self.mueff = weights.sum() ** 2 / (weights**2).sum()

    def ask(self) -> torch.Tensor:
        """Generate new candidate solutions"""
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(self.covariance)
        except torch._C._LinAlgError as e:
            logger.error(traceback.format_exc())
            logger.error(self.covariance)
            raise e
        samples = []
        for _ in range(self.population_size):
            z = torch.randn(self.dimension, device=self.device)
            x = self.mean + self.sigma * (eigenvectors @ (torch.sqrt(eigenvalues) * z))
            samples.append(x)
        return torch.stack(samples)

    def tell(self, solutions: torch.Tensor, fitness_values: torch.Tensor):
        """Update the internal model using the evaluated solutions"""
        if torch.any(torch.isnan(solutions)):
            return

        # Sort by fitness
        sorted_indices = torch.argsort(fitness_values)
        sorted_solutions = solutions[sorted_indices]

        # Compute weighted mean
        old_mean = self.mean.clone()
        self.mean = (
            self.weights.unsqueeze(1) * sorted_solutions[: self.population_size]
        ).sum(dim=0)

        # Update evolution paths
        y = self.mean - old_mean
        z = y / self.sigma

        cs_term = torch.tensor(self.cs * (2 - self.cs) * self.mueff, device=self.device)
        self.ps = (1 - self.cs) * self.ps + torch.sqrt(cs_term) * z

        hsig_denom = 1 - (1 - self.cs) ** (2 * self.population_size)
        hsig = self.ps.norm() / torch.sqrt(
            torch.tensor(hsig_denom, device=self.device)
        ) < (1.4 + 2 / (self.dimension + 1))

        cc_term = torch.tensor(self.cc * (2 - self.cc) * self.mueff, device=self.device)
        self.pc = (1 - self.cc) * self.pc + hsig * torch.sqrt(cc_term) * y

        # Update covariance matrix
        rank_one = self.pc.outer(self.pc)
        rank_mu = sum(
            self.weights[i]
            * (sorted_solutions[i] - old_mean).outer(sorted_solutions[i] - old_mean)
            for i in range(self.population_size)
        ) / (self.sigma**2)

        self.covariance = (
            (1 - self.c1 - self.cmu) * self.covariance
            + self.c1 * rank_one
            + self.cmu * rank_mu
        )

        # Update step size
        self.sigma *= torch.exp(
            (self.cs / self.damps)
            * (self.ps.norm() / torch.sqrt(torch.tensor(self.dimension)) - 1)
        ).item()

        self.covariance += torch.eye(self.dimension, device=self.device) * 0.01
        if self.sigma < 0.01:
            self.sigma += 0.01


def example_function(x: torch.Tensor) -> float:
    """Simple test function (sphere function)"""
    h = x - torch.arange(x.shape[0], device=x.device)
    return (h**2).sum()


# Example usage
if __name__ == "__main__":
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize optimizer
    dimension = 10
    optimizer = CMAES(dimension=dimension, device=device)

    # Run optimization
    num_generations = 100
    best_fitness = float("inf")

    for generation in range(num_generations):
        # Generate new solutions
        solutions = optimizer.ask()

        # Evaluate solutions
        fitness_values = torch.tensor(
            [example_function(x) for x in solutions], device=device
        )

        # Update best fitness
        generation_best = fitness_values.min().item()
        if generation_best < best_fitness:
            best_fitness = generation_best

        # Update internal model
        optimizer.tell(solutions, fitness_values)

        # Print progress
        if generation % 10 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness:.6f}")
            print(f"Mean = {optimizer.mean}")
            print(f"Step size (sigma) = {optimizer.sigma}\n")

        # Check for convergence
        if optimizer.sigma < 1e-8:
            print("Converged!")
            break

    print(f"\nFinal solution:")
    print(f"Best fitness: {best_fitness:.6f}")
    print(f"Solution: {optimizer.mean}")
