import strawberryfields as sf
import strawberryfields.ops as ops
import numpy as np

M = 4
alpha = 2
r = 1.55
coh_ind = 2

def get_gauss(M, gate_name, r):
    eng = sf.Engine(backend='gaussian') # The engine cannot be reused.
    prog = sf.Program(M)
    with prog.context as q:
        for i in range(0, M):
            if gate_name == 'Coherent':
                ops.Coherent(r=r) | q[i]
                # ops.Coherent(r=r) | q[i]
            elif gate_name == 'Dgate':
                ops.Dgate(r=r) | q[i]
                # ops.Dgate(r=r) | q[i]
            elif gate_name == 'Squeezed':
                ops.Squeezed(r=r) | q[i]
            elif gate_name == 'Sgate':
                ops.Sgate(r=r) | q[i]
            else:
                raise Exception('Gate not supported')

    state = eng.run(prog).state

    mu = state.means()
    cov = state.cov()

    return mu, cov


mu_1, cov_1 = get_gauss(M, 'Sgate', alpha)
mu_2, cov_2 = get_gauss(M, 'Squeezed', alpha)

print('mu matrix')
print(mu_1)
print(mu_2)
print('The mu matrices are the same: {}'.format(np.allclose(mu_1, mu_2)))

print('cov matrix')
print(cov_1)
print(cov_2)
print('The cov matrices are the same: {}'.format(np.allclose(cov_1, cov_2)))


