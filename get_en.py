import numpy as np
class Generate_entangled_state(object):
    def __init__(self, name='ent_test_set', size=200, sub_dim=2, space_number=3):
        self.name = name
        self.size = size
        self.sub_dim = sub_dim
        self.space_number = space_number
        self.v1 = [1, 0, 0, 0, 0, 0, 0, 0]
        self.v2 = [0, 0, 0, 0, -0.5, -0.5, 0.5, 0.5]
        self.v3 = [0, 0, -0.5, 0.5, 0, 0, -0.5, 0.5]
        self.v4 = [0, -0.5, 0, -0.5, 0, 0.5, 0, 0.5]
        p1 = np.outer(self.v1, self.v1)
        p2 = np.outer(self.v2, self.v2)
        p3 = np.outer(self.v3, self.v3)
        p4 = np.outer(self.v4, self.v4)
        self.ptile = (np.eye(8) - p1 - p2 - p3 - p4) / 4
        self.state = []

    def generate_sub_matrix(self):
        q1 = np.random.normal(0, 1, [self.sub_dim, self.sub_dim])
        q2 = np.random.normal(0, 1, [self.sub_dim, self.sub_dim])
        q = q1 + 1j * q2
        q = np.matrix(q)  # Create a random Complex matrix H.
        q = (q + q.H) / 2  # Generate GUE
        q = np.dot(q, q.H) / np.trace(np.dot(q, q.H))  # Make it positive, and normalize to unity.
        return q

    def generate(self):
        for s in range(self.size):
            sigma = []
            for i in range(self.space_number):
                q1 = np.random.normal(0, 1, [self.sub_dim, self.sub_dim])
                q2 = np.random.normal(0, 1, [self.sub_dim, self.sub_dim])
                q = q1 + 1j * q2
                sigma.append(q)  # Create a random Complex matrix H.
            Sigma = np.reshape(np.einsum('ij,kl,mn->ikmjln', sigma[0], sigma[1], sigma[2]),
                               [pow(self.sub_dim, self.space_number), pow(self.sub_dim, self.space_number)])
            Sigma = np.matrix(Sigma)
            transition = np.dot(np.dot(Sigma, self.ptile), Sigma.H)
            rho = transition / np.trace(transition)
            self.state.append(rho)
        State = np.array(self.state)
        shape = list(State.shape)
        shape.append(1)
        Set_r = np.reshape(np.real(State), shape)
        Set_i = np.reshape(np.imag(State), shape)
        Set_2 = np.concatenate((Set_r, Set_i), axis=-1)
        np.save('./pptmixer/' + self.name + '.npy', Set_2)
Generate_entangled_state(name='3-qubit_ent',size=2000).generate()
en=np.load('./pptmixer/3-qubit_ent.npy')
for (i,x,y) in zip(np.arange(len(en)),en[:,:,:,0],en[:,:,:,1]): 
    np.savetxt('./pptmixer/3-qubit_ent/'+'state_real'+repr(i)+'.txt',x, fmt="%.10f",delimiter=" ")
    np.savetxt('./pptmixer/3-qubit_ent/'+'state_imag'+repr(i)+'.txt',y, fmt="%.10f",delimiter=" ")

 np.savetxt('./pptmixer/3-qubit_ent/ptile.txt',ptile, fmt="%.10f",delimiter=" ")
