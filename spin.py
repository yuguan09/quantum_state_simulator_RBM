import numpy as np

class RBM():
    """
    A class used to represent a Restricted Boltzmann Machine (RBM)
    ...

    Attributes
    ----------
    num_visible : int
    Number of visible units
    num_hidden : int
    Number of hidden units
    visible_biases : array-like
    Bias values for the visible layer
    hidden_biases : array-like
    Bias values for the hidden layer
    weights : array-like
    Weights between the visible and hidden layer units

    Methods
    -------
    wave_function(visible_state)
    Compute the wave function for a given visible state
    """
    def __init__(self, num_visible, num_hidden):
        """
             Parameters
             ----------
             num_visible : int
             Number of visible units
             num_hidden : int
             umber of hidden units
        """
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.visible_biases = np.random.randn(num_visible)
        self.hidden_biases = np.random.randn(num_hidden)
        self.weights = np.random.randn(num_visible, num_hidden)

    def wave_function(self, visible_state):
        """
        Compute the wave function for a given visible state

        Parameters
        ----------
        visible_state : array-like
            The visible state

        Returns
        -------
        float
            The wave function value for the given visible state
        """
        visible_term = np.exp(np.dot(visible_state, self.visible_biases))
        hidden_term = np.prod(2 * np.cosh(np.dot(visible_state, self.weights) + self.hidden_biases))

        return visible_term * hidden_term

    def sample(self, visible_state, num_steps=1):
        """
        Perform Gibbs sampling starting from a given visible state.

        Parameters
        ----------
        visible_state : array-like
            The visible state to start the Gibbs sampling from
        num_steps : int, optional
            The number of Gibbs sampling steps to perform (default is 1)

        Returns
        -------
        array-like
            The new visible state after Gibbs sampling
        """
        for _ in range(num_steps):
            hidden_probs = sigmoid(np.dot(visible_state, self.weights) + self.hidden_biases)
            hidden_state = np.random.binomial(1, hidden_probs)

            visible_probs = sigmoid(np.dot(hidden_state, self.weights.T) + self.visible_biases)
            visible_state = np.random.binomial(1, visible_probs)

            return visible_state

    def local_energy_Ising(self, state, h):
        num_state = len(state)
        hamiltonian_value = Hamitonian_Ising(state, h)
        psi = self.wave_function(state)

        local_energy = -hamiltonian_value * psi
        return local_energy

    def SDG_opti(self, alpha=1, step_nums = 40):
        pass

    def mh_sample(self, state, h, beta = 1.0):
        N = len(state)
        new_state = state.copy()

        spin_to_filp = np.random.randint(N)
        new_state[spin_to_filp] *= -1

        energy_diff = Hamitonian_Ising(new_state, h) - Hamitonian_Ising(state, h)
        psi_ratio_squared = np.abs(self.wave_function(new_state) / self.wave_function(state)) ** 2
        acceptance_probability = min(1, np.exp(-beta * energy_diff) * psi_ratio_squared)

        if np.random.rand() < acceptance_probability:
            return new_state
        else:
            return state

    def train(self, initial_state, h, lr = 0.01, num_iteration = 1000, burn_in = 100, num_samples = 1000, beta = 1.0):
        N, M = self.weights.shape
        state = initial_state.copy()
        for iteration in range(num_iteration):
            local_energies = []
            spin_states = []

            for _ in range(num_samples + burn_in):
                state = self.mh_sample(state, h, beta)
                if _ >= burn_in:
                    local_energy = self.local_energy_Ising(state, h)
                    local_energies.append(local_energy)
                    spin_states.append(state.copy())

            # Calculate the gradient of a, b, w
            a_deriv = np.zeros(N, dtype=np.float64)
            w_deriv = np.zeros((N, M), dtype=np.float64)
            b_deriv = np.zeros(M, dtype= np.float64)

            for state, local_energy in zip(spin_states, local_energies):
                psi_ratio = self.wave_function(state) / self.wave_function(initial_state)
                factor = -h * local_energy * psi_ratio

                # Derivative of a
                a_deriv += factor * 2 * state

                # Derivative of W
                w_deriv += factor * np.outer(state, np.tanh(self.effective_angles(state)))

                # Derivative of b
                b_deriv += factor * (1 - np.tanh(self.effective_angles(state)) ** 2)

                # Update the parameters using the gradient and the learning rate
            self.visible_biases += lr * a_deriv / num_samples
            self.weights += lr * w_deriv / num_samples
            self.hidden_biases += lr * b_deriv / num_samples

def sigmoid(x):
    """
    Calculate the sigmoid activation of a given input x

    Parameters:
        x (float or array-like): Input to the sigmoid function

    Returns:
        float or array-like: Output from the sigmoid function
    """
    return 1 / (1 + np.exp(-x))

def Hamitonian_Ising(spin_state, h):
    """
    calculate the hamiltonian of the given spin state for an Ising model.

    Args:
        spin_state(): The Ising spin state.

    Returns:
        float: The hamiltonian of the spin state.
    """

    h = h
    J = 1
    N = len(spin_state)

    field_interaction = -h * np.sum(spin_state)
    spin_interaction = -J * np.sum(spin_state[i] * spin_state[(i + 1) % N] for i in range(N))

    return field_interaction + spin_interaction

def Ising_energy_expection(spin_state, rbm, num_samples = 300):
    """
    Estimate the energy expectation of the given spin state using Monte Carlo sampling.

    Args:
    spin_state (numpy array): The initial spin state.
    rbm (RBM object): The RBM object.
    num_samples (int): The number of samples to draw.

    Returns:
    float: The estimated energy expectation.
    """
    energy_sum = 0.0
    probability_sum = 0.0

    for _ in range(num_samples):
        new_spin_state = rbm.sample(spin_state)
        energy = Hamitonian_Ising(new_spin_state)
        probability = np.abs(rbm.wave_function(new_spin_state))**2

        energy_sum += energy * probability
        probability_sum += probability

        spin_state = new_spin_state

    return energy_sum / probability_sum

def energy_expectation(spin_state, rbm, num_samples=10000):
    energy_sum = 0.0
    probability_sum = 0.0
    warm_up = num_samples // 2

    for i in range(num_samples):
        # Sample a new spin state
        new_spin_state = rbm.sample(spin_state)

        # Calculate the probability ratio of the new and old spin states
        old_prob = np.abs(rbm.wavefunction(spin_state))**2
        new_prob = np.abs(rbm.wavefunction(new_spin_state))**2
        prob_ratio = new_prob / old_prob

        # Accept or reject the new spin state
        if np.random.random() < prob_ratio:
            spin_state = new_spin_state

        # Only consider the second half of the samples to ensure convergence
        if i >= warm_up:
            # Calculate the Hamiltonian of the spin state
            energy = Hamitonian_Ising(spin_state)

            # Calculate the probability of the spin state
            probability = np.abs(rbm.wavefunction(spin_state))**2

            energy_sum += energy * probability
            probability_sum += probability

    return energy_sum / probability_sum

def Quantum_Monte_Carlo_Simulation(spin_state, rbm, num_samples=10000):
    pass


if __name__ == '__main__':
    spin_state = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    energy = Hamitonian_Ising(spin_state)
    print(energy)
    print("yy\n")

    rbm_state = RBM(len(spin_state), 5)
    energy_expectation = Ising_energy_expection(spin_state, rbm_state)
    print(energy_expectation)
