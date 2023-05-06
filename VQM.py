import numpy as np

def random_complex(size):
    a = (np.random.random(size) - .5) * 10e-2
    b = (np.random.random(size) - .5) * 10e-2
    return a + 1j*b

N = 4
alpha = 2
M = alpha * N
hfield = 2  #?
a = random_complex(N)
b = random_complex(M)
w = random_complex(N, M)
state = np.random.randint(2, size=N)
state[state == 0] = -1
state_i = list(range(N))

def effective_angles(state):
    return b+np.inner(np.transpose(W), state)

def Psi_M(state, a, b, w):
    return np.exp(np.inner(a, state)) * np.prod(2 * np.cosh(effective_angles(state)))

def E_loc(state):
    E = 0
    for i in state_i:
        if i == N-1:
            E -= (state[i]*state[0])
        else:
            E -= (state[i] * state[i + 1])

        Psi_M_s = Psi_M(state, a, b, w)
        for i in state_i:
            state[i] *= -1
            E -= hfield*Psi_M(state, a, b, w)/Psi_M_s
            state[i] *= -1

        return E/N

def step():
    sites = np.random.choice(state_i, 1)
    Psi_M_before = Psi_M(state, a, b, w)
    for i in sites:
        state[i] *= -1
    Psi_M_after = Psi_M(state, a, b, w)
    acceptance = np.real(Psi_M_after*np.conj(Psi_M_after)/(Psi_M_before*np.conj(Psi_M_before)))

    if acceptance < np.random.uniform():
        for i in sites:
            state[i] *= -1
        return 1
    else:
        return 0


block_E = []
for block_i in range(30):
    state = np.random.randint(2, size=N)
    state[state == 0] = -1
    for k in range(10000):
        step()

    iterations = 20000
    rejected = 0
    array_E_loc = []

    array_a_d = []
    array_b_d = []
    array_w_d = []

    for k in range(iterations):
        rejected += step()

        if k % 100 == 0:
            Psi_M_s = Psi_M(state, a, b, W)

            # Derivative a
            a_deriv = np.zeros(N, dtype=np.complex_)
            for i in range(N):
                state[i] *= -1  # flip
                a_deriv[i] = -hfield * Psi_M(state, a, b, W) / Psi_M_s * 2. * state[i]
                state[i] *= -1  # flip back

            # Derivative W
            dW = np.zeros((N, M), dtype=np.complex_)
            for w_i in range(N):
                for w_j in range(M):
                    dw_sum = 0
                    before_flip = np.tanh(effective_angles(state))
                    for i in range(N):
                        state[i] *= -1  # flip
                        dw_sum += Psi_M(state, a, b, W) / Psi_M_s * (
                                -state[i] * np.tanh(effective_angles(state)[w_j]) - state[i] * before_flip[w_j])
                        state[i] *= -1  # flip back
                    dw_sum *= hfield
                    dW[w_i, w_j] = dw_sum

            # Derivative b
            b_deriv = np.zeros(M, dtype=np.complex_)
            for b_j in range(M):
                tanh_before_flip = np.tanh(effective_angles(state))
                db_sum = 0
                for i in range(N):
                    state[i] *= -1  # flip
                    db_sum += Psi_M(state, a, b, W) / Psi_M_s * (
                            np.tanh(effective_angles(state)[b_j]) - tanh_before_flip[b_j])
                    state[i] *= -1  # flip back
                b_deriv[b_j] = -hfield * db_sum

            array_a_d.append(a_deriv)
            array_b_d.append(b_deriv)
            array_w_d.append(dW)
            array_E_loc.append(np.real(E_loc(state)))

    print('%d. E_loc=%.4f std=%.4f (%.1f %% moves rejected)' % (block_i + 1,
                                                                np.mean(array_E_loc),
                                                                np.std(array_E_loc) / (np.sqrt(len(array_E_loc))),
                                                                100. * rejected / iterations))

    block_E.append(np.mean(array_E_loc))
    mean_da = np.mean(np.array(array_a_d), axis=0)
    mean_db = np.mean(np.array(array_b_d), axis=0)
    mean_dw = np.mean(np.array(array_w_d), axis=0)
    a = a - .1 * mean_da
    b = b - .1 * mean_db
    W = W - .1 * mean_dw
