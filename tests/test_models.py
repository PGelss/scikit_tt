# -*- coding: utf-8 -*-

import numpy as np
import scikit_tt.tensor_train as tt
import scikit_tt.models as mdl
from unittest import TestCase


class TestModels(TestCase):

    def setUp(self):
        """Setup for testing models"""

        # define parameters
        # -----------------

        # set tolerance
        self.tol = 1e-1

        # set order
        self.order = 3

        # set CO adsorption rate for CO oxidation model
        self.k_ad_co = 1e4

        # set fequencies for kuramoto model
        self.frequencies = np.linspace(-5, 5, self.order)

        # set parameters for two-step destruction
        self.k_1 = 1
        self.k_2 = 2
        self.k_3 = 1
        self.m = 3

        # set parameters for toll station model
        self.number_of_lanes = 20
        self.number_of_cars = 10

        # generate Cantor dusts
        # ---------------------

        # generate 1-dimensional Cantor dust of level 3
        generator = np.array([1, 0, 1])
        self.cantor_dust = [np.kron(np.kron(generator, generator), generator)]

        # generate 2-dimensional Cantor dust of level 3
        generator = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
        self.cantor_dust.append(np.kron(np.kron(generator, generator), generator))

        # generate 3-dimensional Cantor dust of level 3
        generator = np.array(
            [[[1, 0, 1], [0, 0, 0], [1, 0, 1]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[1, 0, 1], [0, 0, 0], [1, 0, 1]]])
        self.cantor_dust.append(np.kron(np.kron(generator, generator), generator))

        # generate multisponges
        # ---------------------

        # generate 2-dimensional multisponge of level 3
        generator = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        self.multisponge = [np.kron(np.kron(generator, generator), generator)]

        # generate 3-dimensional multisponge of level 3
        generator = np.array(
            [[[1, 1, 1], [1, 0, 1], [1, 1, 1]], [[1, 0, 1], [0, 0, 0], [1, 0, 1]], [[1, 1, 1], [1, 0, 1], [1, 1, 1]]])
        self.multisponge.append(np.kron(np.kron(generator, generator), generator))

        # generate RGB fractal
        # --------------------

        # define RGB matrices
        self.matrix_r = np.random.rand(3, 3)
        self.matrix_g = np.random.rand(3, 3)
        self.matrix_b = np.random.rand(3, 3)

        # generate RGB fractal
        self.rgb_fractal = np.zeros([27, 27, 3])
        self.rgb_fractal[:, :, 0] = np.kron(np.kron(self.matrix_r, self.matrix_r), self.matrix_r)
        self.rgb_fractal[:, :, 1] = np.kron(np.kron(self.matrix_g, self.matrix_g), self.matrix_g)
        self.rgb_fractal[:, :, 2] = np.kron(np.kron(self.matrix_b, self.matrix_b), self.matrix_b)

        # generate VICsek fractals
        # ------------------------

        # generate 2-dimensional Vicsek fractal of level 3
        generator = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        self.vicsek_fractal = [np.kron(np.kron(generator, generator), generator)]

        # generate 3-dimensional Vicsek fractal of level 3
        generator = np.array(
            [[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
        self.vicsek_fractal.append(np.kron(np.kron(generator, generator), generator))
        
        # quantum models
        # --------------
        
        self.H_quantum = 1/np.sqrt(2) * np.array([[1,1],[1,-1]])
        self.C0_quantum = np.array([[1,0],[0,0]])
        self.C1_quantum = np.array([[0,0],[0,1]])
        self.A_quantum = self.H_quantum @ self.C0_quantum @ self.H_quantum
        self.B_quantum = self.H_quantum @ self.C1_quantum @ self.H_quantum
        self.NOT_quantum = np.array([[0,1],[1,0]])
        self.CNOT_quantum = np.kron(self.C0_quantum, np.eye(2)) + np.kron(self.C1_quantum, self.NOT_quantum)
        
        # QFA
        self.qfa_mat = np.zeros([16,16])
        self.qfa_mat[0:2,0:2] = np.eye(2)
        self.qfa_mat[8:10,8:10] = np.eye(2)
        self.qfa_mat[10:12,2:4] = np.eye(2)
        self.qfa_mat[12:14,4:6] = np.eye(2)
        self.qfa_mat[6:8,6:8] = self.NOT_quantum
        self.qfa_mat[14:16,14:16] = self.NOT_quantum
        self.qfa_mat[2:4,10:12] = self.NOT_quantum
        self.qfa_mat[4:6,12:14] = self.NOT_quantum
        
        # QFAN
        self.number_of_adders = 3
        self.qfan_mat = np.kron(np.eye(64), self.qfa_mat)@np.kron(np.kron(np.eye(8), self.qfa_mat), np.eye(8))@np.kron(self.qfa_mat, np.eye(64))
        
        # Simon's algorithm
        self.simon_G1 = np.kron(self.H_quantum, np.eye(2))
        self.simon_G1 = np.kron(self.simon_G1, self.simon_G1)
        self.simon_G1 = np.kron(self.simon_G1, self.simon_G1)
        self.simon_G2 = np.kron(np.eye(64),self.CNOT_quantum)
        self.simon_G2 = self.simon_G2 @ np.kron(np.kron(np.eye(16),self.CNOT_quantum), np.eye(4))
        self.simon_G2 = self.simon_G2 @ np.kron(np.kron(np.eye(4),self.CNOT_quantum), np.eye(16))
        self.simon_G2 = self.simon_G2 @ np.kron(self.CNOT_quantum, np.eye(64))
        self.simon_G3 = np.kron(self.C0_quantum, np.eye(128)) + np.kron(np.kron(self.C1_quantum, np.eye(16)), np.kron(self.NOT_quantum,np.eye(4)))
        self.simon_G3 = self.simon_G3 @ np.kron(self.CNOT_quantum, np.eye(64))
        self.simon_mat = self.simon_G1 @ self.simon_G3 @ self.simon_G2 @ self.simon_G1
        self.simon_state = self.simon_mat @ np.eye(256)[:,0]
        
        # (inverse) quantum Fourtier transform
        self.qft_n = 4
        self.qft_m = 2**self.qft_n
            
        # Shor's algorithm
        self.shor_M = 15
        self.shor_a = [2,4,7,8]
        self.shor_n = 4
        self.shor_m = 2**(2*self.shor_n)
        
    def test_qfa(self):
        """test for QFA"""

        # construct TT representation of QFA
        qfa = mdl.qfa()
                
        # check if construction is correct
        self.assertEqual(np.linalg.norm(qfa.matricize()-self.qfa_mat), 0)
        
    def test_qfan(self):
        """test for QFAN"""

        # construct TT representation of QFAN
        qfan = mdl.qfan(self.number_of_adders)
                
        # check if construction is correct
        self.assertEqual(np.linalg.norm(qfan.matricize()-self.qfan_mat), 0)
        
    def test_simon(self):
        """test for Simon's algorithm"""

        # construct TT representation of Simon's circuit
        simon = mdl.simon()
                
        # check if construction is correct
        self.assertLess(np.linalg.norm(simon.matricize()-self.simon_state), 1e-14)
        
    def test_qft(self):
        """test for Quantum Fourier transform"""
        
        self.qft_basis_vec = np.asarray(np.unravel_index(np.arange(self.qft_m), [2]*self.qft_n)).T
        self.qft_basis_tt = [None]*self.qft_m
        for i in range(self.qft_m):
            self.qft_basis_tt[i] = tt.unit([2]*self.qft_n,self.qft_basis_vec[i,:])
        
        # construct TT representation of QFT
        G = mdl.qft(self.qft_n)
        
        # basis for fractional binary numbers
        frac_basis = np.array([2**(-k) for k in range(1,self.qft_n+1)])
        
        for i in range(self.qft_m):
            
            # construct final states for basis states as inputs
            res_tt = self.qft_basis_tt[i]
            for j in range(self.qft_n):
                res_tt = (G[j]@res_tt).ortho(threshold=1e-12)
                
            # compute fractional binary numbers
            frac_1 = np.exp(2*np.pi*1j*np.inner(self.qft_basis_vec[i,-1:],frac_basis[:1]))
            frac_2 = np.exp(2*np.pi*1j*np.inner(self.qft_basis_vec[i,-2:],frac_basis[:2]))
            
            # compute vector representation of final state
            res_mat = 1/(np.sqrt(2**self.qft_n)) * np.kron([1, frac_2], [1, frac_1])
            for j in range(2,self.qft_n):
                frac = np.exp(2*np.pi*1j*np.inner(self.qft_basis_vec[i,-(j+1):],frac_basis[:(j+1)]))
                res_mat = np.kron([1, frac], res_mat)
    
            # check if construction is correct
            self.assertLess(np.linalg.norm(res_tt.matricize()-res_mat), 1e-14)
            
    def test_iqft(self):
        """test for inverse Quantum Fourier transform"""
        
        self.qft_basis_vec = np.asarray(np.unravel_index(np.arange(self.qft_m), [2]*self.qft_n)).T
        self.qft_basis_tt = [None]*self.qft_m
        for i in range(self.qft_m):
            self.qft_basis_tt[i] = tt.unit([2]*self.qft_n,self.qft_basis_vec[i,:])
        
        # construct TT representation of inverse QFT
        G = mdl.iqft(self.qft_n)
        
        # basis for fractional binary numbers
        frac_basis = np.array([2**(-k) for k in range(1,self.qft_n+1)])
        
        for i in range(self.qft_m):
            
            # construct final states for random basis states as inputs
            res_tt = self.qft_basis_tt[i]
            for j in range(self.qft_n):
                res_tt = (G[j]@res_tt).ortho(threshold=1e-12)
                
            # compute fractional binary numbers
            frac_1 = np.exp(-2*np.pi*1j*np.inner(self.qft_basis_vec[i,-1:],frac_basis[:1]))
            frac_2 = np.exp(-2*np.pi*1j*np.inner(self.qft_basis_vec[i,-2:],frac_basis[:2]))
            
            # compute vector representation of final state
            res_mat = 1/(np.sqrt(2**self.qft_n)) * np.kron([1, frac_2], [1, frac_1])
            for j in range(2,self.qft_n):
                frac = np.exp(-2*np.pi*1j*np.inner(self.qft_basis_vec[i,-(j+1):],frac_basis[:(j+1)]))
                res_mat = np.kron([1, frac], res_mat)
    
            # check if construction is correct
            self.assertLess(np.linalg.norm(res_tt.matricize()-res_mat), 1e-14)
            
    def test_shor(self):
        """test for Shor's algorithm"""
        
        # construct matrix representation of oracle for given values of a
        self.shor_G = [None]*len(self.shor_a)
        for i in range(len(self.shor_a)):
            self.shor_G[i] = np.zeros([2**(3*self.shor_n)]*2)
            # for any j>3: a**j mod 15 = 1
            for j in range(4):
                ind_x = list(np.unravel_index(j, [2]*(2*self.shor_n))) 
                ind_y = list(np.unravel_index(int(np.mod(self.shor_a[i]**j, self.shor_M)), [2]*self.shor_n))
                G_tmp = 1
                for k in range(self.shor_n):
                    G_tmp = np.kron(G_tmp, np.linalg.matrix_power(self.NOT_quantum,ind_y[k]))
                for k in range(2*self.shor_n-1, 2*self.shor_n-3, -1):
                    G_tmp = np.kron(np.diag(np.eye(2)[:,ind_x[k]]), G_tmp)
                G_tmp = np.kron(np.eye(64), G_tmp)
                self.shor_G[i] = self.shor_G[i] + G_tmp
        
        for i in range(len(self.shor_a)):
            
            # construct TT representation
            G_tt = mdl.shor(self.shor_a[i])
            
            # check if construction is correct
            self.assertLess(np.linalg.norm(G_tt.matricize()-self.shor_G[i]), 1e-12)
        
    def test_exciton(self):
        """test for exciton chain"""
        
        # order, excitonic site energy, and coupling strength
        n_site = 5
        alpha=1e-1
        beta=-1e-2
        
        # ladder operators
        raising  = np.diag([1], -1)        
        lowering = np.diag([1], 1)
        
        # matricized SLIM components
        S = alpha * raising @ lowering
        LM = beta*(np.kron(raising, lowering) + np.kron(lowering, raising))
        
        # construct matrix representation
        op_mat = np.zeros([2**n_site, 2**n_site])
        op_mat = op_mat + np.kron(S, np.eye(2**(n_site-1)))
        for i in range(1,n_site-1):
            op_mat = op_mat + np.kron(np.kron(np.eye(2**i), S), np.eye(2**(n_site-1-i)))
        op_mat = op_mat + np.kron(np.eye(2**(n_site-1)), S)
        op_mat = op_mat + np.kron(LM,np.eye(2**(n_site-2)))
        for i in range(1,n_site-1):
            op_mat = op_mat + np.kron(np.kron(np.eye(2**i), LM),np.eye(2**(n_site-2-i)))
            
        op_mat = op_mat + beta * (np.kron(np.kron(raising, np.eye(2**(n_site-2))), lowering) + np.kron(np.kron(lowering, np.eye(2**(n_site-2))), raising))
            
        # construct TT representation
        op = mdl.exciton_chain(n_site, alpha, beta)
        
        # check if construction is correct
        self.assertLess(np.linalg.norm(op.matricize()-op_mat), 1e-14)
        
    def test_cantor_dust(self):
        """test for Cantor dust"""

        # generating Cantor dusts
        cantor_dust = []
        for i in range(3):
            cantor_dust.append(mdl.cantor_dust(i + 1, 3))

        # check if construction is correct
        for i in range(3):
            self.assertEqual(np.sum(self.cantor_dust[i] - cantor_dust[i]), 0)

    def test_multisponge(self):
        """test for multisponge"""

        # generating multisponges
        multisponge = []
        for i in range(2):
            multisponge.append(mdl.multisponge(i + 2, 3))

        # check if construction is correct
        for i in range(2):
            self.assertEqual(np.sum(self.multisponge[i] - multisponge[i]), 0)

        # check if construction fails when dimension equal to 1
        with self.assertRaises(ValueError):
            mdl.multisponge(1, 1)

    def test_rgb_fractal(self):
        """test for rgb_fractal"""

        # generate RGB fractal
        rgb_fractal = mdl.rgb_fractal(self.matrix_r, self.matrix_g, self.matrix_b, 3)

        # check if construction is correct
        self.assertEqual(np.sum(self.rgb_fractal - rgb_fractal), 0)

    def test_vicsek_fractal(self):
        """test for vicsek_fractal"""

        # generating multisponges
        vicsek_fractal = []
        for i in range(2):
            vicsek_fractal.append(mdl.vicsek_fractal(i + 2, 3))

        # check if construction is correct
        for i in range(2):
            self.assertEqual(np.sum(self.vicsek_fractal[i] - vicsek_fractal[i]), 0)

        # check if construction fails when dimension equal to 1
        with self.assertRaises(ValueError):
            mdl.vicsek_fractal(1, 1)

    def test_fpu_kuramoto(self):
        """tests for Fermi-Pasta-Ulam and Kuramoto model"""

        mdl.fpu_coefficients(self.order)
        mdl.kuramoto_coefficients(self.order, self.frequencies)

    def test_co_oxidation(self):
        """tests for CO oxidation"""

        # construct operator
        op = mdl.co_oxidation(self.order, self.k_ad_co, cyclic=True)

        # check if stochastic
        self.assertLess((tt.ones([1]*op.order, op.col_dims).dot(op)).norm(), self.tol)

    def test_signaling_cascade(self):
        """tests for signaling cascade"""

        # construct operator
        op = mdl.signaling_cascade(self.order)

        # check if stochastic
        self.assertLess((tt.ones([1]*op.order, op.col_dims).dot(op)).norm(), self.tol)

    def test_toll_station(self):
        """tests for toll station"""

        # construct operator
        op = mdl.toll_station(self.number_of_lanes, self.number_of_cars)

        # check if stochastic
        self.assertLess((tt.ones([1]*op.order, op.col_dims).dot(op)).norm(), self.tol)

    def test_two_step(self):
        """tests for two-setp destruction"""

        # construct operator
        op = mdl.two_step_destruction(self.k_1, self.k_2, self.k_3, self.m)

        # check if stochastic
        self.assertLess((tt.ones([1]*op.order, op.col_dims).dot(op)).norm(), self.tol)
