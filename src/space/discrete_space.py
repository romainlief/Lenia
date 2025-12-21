from ASpace.space import Space
import torch


class DiscreteSpace(Space):
    """A discrete space in :math:{ 0, 1, ..., n-1 }.
    mutation is gaussian by default: please create custom space inheriting from discrete space for custom mutation functions

    Example:
    >>> DiscreteSpace(2)
    """

    def __init__(self, n, mutation_mean=0.0, mutation_std=1.0, indpb=1.0) -> None:
        assert n >= 0, "n (number of discrete elements) must be non-negative"
        self.n = n
        self.mutation_mean = torch.as_tensor(
            mutation_mean, dtype=torch.float64
        )  # mean of the gaussian mutation
        self.mutation_std = torch.as_tensor(
            mutation_std, dtype=torch.float64
        )  # std of the gaussian mutation
        self.indpb = torch.as_tensor(
            indpb, dtype=torch.float64
        )  # independent probability for each attribute to be mutated
        super(DiscreteSpace, self).__init__((), torch.int64)

    def sample(self):
        """
        Randomly sample an element of this space.
        """
        return torch.randint(self.n, ())

    def mutate(self, x) -> torch.Tensor:
        """
        Randomly mutate an element of this space.
        """
        mutate_mask = torch.rand(self.shape) < self.indpb
        noise = torch.normal(self.mutation_mean, self.mutation_std, ())
        x = x.type(torch.float64) + mutate_mask * noise
        x = torch.floor(x).type(self.dtype)
        if not self.contains(x):
            return self.clamp(x)
        else:
            return x

    def contains(self, x: int | torch.Tensor) -> bool:
        """
        Return boolean specifying if x is a valid member of this space
        """
        if isinstance(x, int):
            as_int = x
        elif not x.dtype.is_floating_point and (x.shape == ()):  # integer or size 0
            as_int = int(x)
        else:
            return False
        return 0 <= as_int < self.n

    def clamp(self, x):
        """
        Return a valid clamped value of x inside space's bounds
        """
        x = torch.max(x, torch.as_tensor(0, dtype=self.dtype, device=x.device))
        x = torch.min(x, torch.as_tensor(self.n - 1, dtype=self.dtype, device=x.device))
        return x

    def __repr__(self):
        """
        String representation of the DiscreteSpace.
        """
        return "DiscreteSpace(%d)" % self.n

    def __eq__(self, other):
        """
        Equality check between two DiscreteSpace instances.
        """
        return isinstance(other, DiscreteSpace) and self.n == other.n
