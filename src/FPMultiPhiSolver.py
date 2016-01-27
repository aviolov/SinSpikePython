# -*- coding:utf-8 -*-
"""
Created on Apr 23, 2012

@author: alex
"""
from __future__ import division

from BinnedSpikeTrain import BinnedSpikeTrain, intervalStats, loadPath
#from Simulator import Path, OUSinusoidalParams
from SpikeTrainSimulator import SpikeTrain, OUSinusoidalParams

from numpy import *
from scipy.sparse import spdiags, lil_matrix
from scipy.sparse.linalg.dsolve.linsolve import spsolve
from copy import deepcopy

from assimulo.solvers.sundials import CVode
from assimulo.problem import Explicit_Problem
from core.numeric import outer
from FPSolver import FIGS_DIR
from InitBox import initialize_right_2std
from Simulator import ABCD_LABEL_SIZE
#from DataHarvester import regime_label

RESULTS_DIR = '/home/alex/Workspaces/Python/OU_hitting_time/Results/FP/'
FIGS_DIR = '/home/alex/Workspaces/Latex/LIFEstimation/Figs/FP'
import os
for D in [RESULTS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)
import time
#from FPSolver import FPSinusoidalSolver
#from Simulator import OUSinusoidalParams

import ext_fpc

class FPMultiPhiSolver():
    def __init__(self, theta, phis, dx, dt, Tf, X_min):  
        self._theta = float(theta) 
        self._phis = phis 
        
        self._v_thresh = 1.
           
        #DISCRETIZATION:
        self.rediscretize(dx, dt, Tf, X_min)
    
    def rediscretize(self, dx, dt, Tf, X_min):
        self._dx = dx
        self._dt = dt

        self._xs = self._space_discretize(X_min)
        self._ts = self._time_discretize(Tf)
    
    def getTf(self):
        return self._ts[-1]
    def setTf(self, Tf):
        self._ts = self._time_discretize(Tf)
    def getXmin(self):
        return self._xs[0]
    def setXmin(self, X_min):
        self._xs = self._space_discretize(X_min)
    
    def _space_discretize(self, X_min):
#        xs = None
#        try:
        xs = arange(self._v_thresh, X_min - self._dx, -self._dx)[-1::-1];
#        except MemoryError:
#            print X_min, self._dx;
        return xs
    
    def _time_discretize(self, Tf):
        return arange(.0, Tf+self._dt, self._dt);

    @classmethod
    def calculate_xmin(cls, Tf, abg, theta):
        #ASSERT max_speed is float and >0.
        alpha, beta, gamma = abg[0], abg[1], abg[2]
        xmin = alpha - abs(gamma)/sqrt(1.0 + theta**2) - 2.0*beta / sqrt(2.0);
        return min([-.25, xmin])
    
    @classmethod
    def calculate_dt(cls, dx, abg, x_min, factor=4.):
        #ASSERT max_speed is float and >0.
        alpha, beta, gamma = abg[0], abg[1], abg[2]
        MAX_SPEED = abs(alpha) + max([abs(x_min), 1.0]) + abs(gamma)   
        return dx / float(MAX_SPEED) * factor; 

    @classmethod
    def calculate_dx(cls, abg, xmin, factor = 1e-1):
        max_speed = abg[0] + abs(abg[2]) - xmin;
        return abg[1] / max_speed * factor;
        
    def _num_nodes(self):
        return len(self._xs)
    def _num_steps (self):
        return len(self._ts)
    def _num_phis(self):
        return len(self._phis)

    def solve(self, abg, visualize=False):
        alpha, beta, gamma = abg[0], abg[1], abg[2]
        theta = self._theta 
        
        dx, dt = self._dx, self._dt;
        xs, ts = self._xs, self._ts;

        #Allocate memory for solution:
        Fs = zeros((self._num_phis(),
                     self._num_steps(),
                      self._num_nodes() ));
        
        #Impose ICs:
#        initial_distribution = ones_like(xs) * (xs > .0);
#        initial_distribution[where(abs(xs) < dx * 1e-4)] = .5;
#        initial_distribution = array(list(initial_distribution)*self._num_phis()).reshape((self._num_phis(),-1)); 
        Fs[:, 0, :] = self._getICs(xs, dx)

        if visualize:
            figure(100);
            for (phis, phi_idx) in zip(self._phis, xrange(self._num_phis())):
                subplot(self._num_phis(), 1, phi_idx+1)
                plot(xs, Fs[phi_idx, 0, :], 'b-*'); 
                title('INITIAL CONDITIONS:');
                xlim((xs[0], xs[-1]) )
                ylim((-.2, 1.2))             
        
        #Solve it using C-N/C-D:
        D = beta * beta / 2.; #the diffusion coeff
        
        #AlloCATE mass mtx:    
        M = lil_matrix((self._num_nodes() - 1, self._num_nodes() - 1));
        e = ones(self._num_nodes() - 1);
        
        dx_sqrd = dx * dx;
        
        d_on = D * dt / dx_sqrd;
        centre_diag = e + d_on;
        centre_diag[-1] = 1.0;
        M.setdiag(centre_diag)

        for tk in xrange(1, self._num_steps()):
            t_prev = ts[tk-1]
            t_next = ts[tk]
#            
            for phi, phi_idx in zip(self._phis, xrange(0, self._num_phis() )):
                #Rip the previous time solution:
                F_prev = Fs[phi_idx, tk - 1, :];
                if max(abs(F_prev)) < 1e-5:
                    continue
                 
                #Advection coefficient:
                U_prev = -(alpha - xs + gamma * sin(theta * (t_prev + phi)));
                U_next = -(alpha - xs + gamma * sin(theta * (t_next + phi)));
        
                #Form the right hand side:
                L_prev = U_prev[1:] * r_[ ((F_prev[2:] - F_prev[:-2]) / 2. / dx,
                                               (F_prev[-1] - F_prev[-2]) / dx)] + \
                                D * r_[(diff(F_prev, 2),
                                           - F_prev[-1] + F_prev[-2])] / dx_sqrd; #the last term comes from the Neumann BC:
                RHS = F_prev[1:] + .5 * dt * L_prev;
                #impose the right BCs:
                RHS[-1] = 0.;
    
                #Reset the 'mass' matrix:
                flow = .5 * dt / dx / 2. * U_next[1:];
                
                d_off = -.5 * D * dt / dx_sqrd;
    #            d_on = D * dt / dx_sqrd;
                
                #With Scipy .11 we have the nice diags function:
                #TODO: local Peclet number should determine whether we central diff, or upwind it! 
                L_left = d_off + flow;
                M.setdiag(r_[(L_left[1:-1], -1.0)], -1);
                
                L_right = d_off - flow;
                M.setdiag(r_[(L_right[:-1])], 1);
                
                #Thomas Solve it:
                Mx = M.tocsr()
                F_next = spsolve(Mx, RHS);
                if visualize:
                    if rand() < 1./ (1+ log(self._num_steps())):
                        figure()
                        plot(xs[1:], F_next); title('t=' + str(t_next) + ' \phi = %.2g'%phi);
                    
                #Store solution:
                Fs[phi_idx, tk, 1:] = F_next;  
            
            #Break out of loop?
            if amax(Fs[:, tk, :]) < 1e-5:
                break   

        #Return:
        return Fs

    def c_solve(self, abg):
        #calls a C routine to solve the PDE (using the same algo as in solve): 
        abgth = r_[abg, self._theta];
        phis = array(self._phis);
        ts = self._ts;
#        xs = self._xs; //I don't have a clue why, but this form of xs breaks the routine, while the one below does not!!!
        xs = linspace(self._xs[0], self._xs[-1], len(self._xs))    
        #TODO: enforce that all params passed donw are NUMPY ARRAYS
        Fs = ext_fpc.solveFP(abgth,
                             phis, ts, xs)
        
            
        return Fs;

    def solve_tau(self, abgt, visualize=False):
        alpha, beta, gamma, tau = abgt[0], abgt[1], abgt[2], abgt[3]
        theta = self._theta 
        
        dx, dt = self._dx, self._dt;
        xs, ts = self._xs, self._ts;

        #Allocate memory for solution:
        Fs = zeros((self._num_phis(),
                     self._num_steps(),
                      self._num_nodes() ));
        
        #Impose ICs:
        Fs[:, 0, :] = self._getICs(xs, dx)

        if visualize:
            figure(100);
            for (phis, phi_idx) in zip(self._phis, xrange(self._num_phis())):
                subplot(self._num_phis(), 1, phi_idx+1)
                plot(xs, Fs[phi_idx, 0, :], 'b-*'); 
                title('INITIAL CONDITIONS:');
                xlim((xs[0], xs[-1]) )
                ylim((-.2, 1.2))             
        
        #Solve it using C-N/C-D:
        D = beta * beta / 2.; #the diffusion coeff
        
        #AlloCATE mass mtx:    
        M = lil_matrix((self._num_nodes() - 1, self._num_nodes() - 1));
        e = ones(self._num_nodes() - 1);
        
        dx_sqrd = dx * dx;
        
        d_on = D * dt / dx_sqrd;
        centre_diag = e + d_on;
        centre_diag[-1] = 1.0;
        M.setdiag(centre_diag)

        for tk in xrange(1, self._num_steps()):
            t_prev = ts[tk-1]
            t_next = ts[tk]
#            
            for phi, phi_idx in zip(self._phis, xrange(0, self._num_phis() )):
                #Rip the previous time solution:
                F_prev = Fs[phi_idx, tk - 1, :];
                if max(abs(F_prev)) < 1e-5:
                    continue
                 
                #Advection coefficient:
                U_prev = -(alpha - xs/tau + gamma * sin(theta * (t_prev + phi)));
                U_next = -(alpha - xs/tau + gamma * sin(theta * (t_next + phi)));
        
                #Form the right hand side:
                L_prev = U_prev[1:] * r_[ ((F_prev[2:] - F_prev[:-2]) / 2. / dx,
                                               (F_prev[-1] - F_prev[-2]) / dx)] + \
                                D * r_[(diff(F_prev, 2),
                                           - F_prev[-1] + F_prev[-2])] / dx_sqrd; #the last term comes from the Neumann BC:
                RHS = F_prev[1:] + .5 * dt * L_prev;
                #impose the right BCs:
                RHS[-1] = 0.;
    
                #Reset the 'mass' matrix:
                flow = .5 * dt / dx / 2. * U_next[1:];
                
                d_off = -.5 * D * dt / dx_sqrd;
    #            d_on = D * dt / dx_sqrd;
                
                #With Scipy .11 we have the nice diags function:
                #TODO: local Peclet number should determine whether we central diff, or upwind it! 
                L_left = d_off + flow;
                M.setdiag(r_[(L_left[1:-1], -1.0)], -1);
                
                L_right = d_off - flow;
                M.setdiag(r_[(L_right[:-1])], 1);
                
                #Thomas Solve it:
                Mx = M.tocsr()
                F_next = spsolve(Mx, RHS);
                if visualize:
                    if rand() < 1./ (1+ log(self._num_steps())):
                        figure()
                        plot(xs[1:], F_next); title('t=' + str(t_next) + ' \phi = %.2g'%phi);
                    
                #Store solution:
                Fs[phi_idx, tk, 1:] = F_next;  
            
            #Break out of loop?
            if max(Fs[:, tk, :]) < 1e-5:
                break   

        #Return:
        return Fs

    def solveFphi(self, abg, Fs, visualize=False):
        alpha, beta, gamma = abg[0], abg[1], abg[2]
        theta = self._theta 
        
        dx, dt = self._dx, self._dt;
        xs, ts = self._xs, self._ts;

        #Allocate memory for solution:
        Fphis = zeros((self._num_phis(),
                     self._num_steps(),
                      self._num_nodes() ));
        
        #Impose ICs:
#        Fs[:, 0, :] = zeros - automatic

        if visualize:
            figure(100);
            for (phis, phi_idx) in zip(self._phis, xrange(self._num_phis())):
                subplot(self._num_phis(), 1, phi_idx+1)
                plot(xs, Fphis[phi_idx, 0, :], 'b-*'); 
                title('INITIAL CONDITIONS:');
                xlim((xs[0], xs[-1]) )
                ylim((-.2, 1.2))             
        
        #Solve it using C-N/C-D:
        D = beta * beta / 2.; #the diffusion coeff
        
        #AlloCATE mass mtx:    
        M = lil_matrix((self._num_nodes() - 1, self._num_nodes() - 1));
        e = ones(self._num_nodes() - 1);
        
        dx_sqrd = dx * dx;
        
        d_on = D * dt / dx_sqrd;
        centre_diag = e + d_on;
        centre_diag[-1] = 1.0;
        M.setdiag(centre_diag)

        for tk in xrange(1, self._num_steps()):
            t_prev = ts[tk-1]
            t_next = ts[tk]
#            
            for phi, phi_idx in zip(self._phis, xrange(0, self._num_phis() )):
                #Rip the previous time solution:
                Fphi_prev = Fphis[phi_idx, tk - 1, :];
                F_next = Fs[phi_idx, tk - 1, :];
                if amax(abs(F_next)) < 1e-4 and amax(abs(Fphi_prev)) < 1e-5:
                    continue
                 
                #Advection coefficient:
                U_prev = -(alpha - xs + gamma * sin(theta * (t_prev + phi)));
                U_next = -(alpha - xs + gamma * sin(theta * (t_next + phi)));
        
                #Form the right hand side:
                L_prev = U_prev[1:] * r_[ ((Fphi_prev[2:] - Fphi_prev[:-2]) / 2. / dx,
                                               (Fphi_prev[-1] - Fphi_prev[-2]) / dx)] + \
                                D * r_[(diff(Fphi_prev, 2),
                                           - Fphi_prev[-1] + Fphi_prev[-2])] / dx_sqrd; #the last term comes from the Neumann BC:
                F_source = gamma * theta * cos(theta* (t_next + phi) ) * diff(F_next)/dx;
                RHS = Fphi_prev[1:] + dt *(.5 * L_prev + F_source);
                
                #impose the right BCs:
                RHS[-1] = 0.;
    
                #Reset the 'mass' matrix:
                flow = .5 * dt / dx / 2. * U_next[1:];
                
                d_off = -.5 * D * dt / dx_sqrd;
    #            d_on = D * dt / dx_sqrd;
                
                #With Scipy .11 we have the nice diags function:
                #TODO: local Peclet number should determine whether we central diff, or upwind it! 
                L_left = d_off + flow;
                M.setdiag(r_[(L_left[1:-1], -1.0)], -1);
                
                L_right = d_off - flow;
                M.setdiag(r_[(L_right[:-1])], 1);
                
                #Thomas Solve it:
                Mx = M.tocsr()
                Fphi_next = spsolve(Mx, RHS);
                if visualize:
                    if rand() < 1./ (1+ log(self._num_steps())):
                        figure()
                        plot(xs[1:], Fphi_next); title('t=' + str(t_next) + ' \phi = %.2g'%phi);
                    
                #Store solution:
                Fphis[phi_idx, tk, 1:] = Fphi_next;  
            
            #Break out of loop?
            if amax(Fs[:, tk, :]) < 1e-5:
                break   

        #Return:
        #TODO: big todo (missing minus sign!!!!:
        return -Fphis
    
    def solveAdjoint(self, abg, Ls, visualize=False):
        ''' abg are the params, 
        Ls is the deviation from the data (Fs|_\vt - S )(t)  of the solution of the forward problem,
        ...
        We are solving BACKWARDS!!!'''
        alpha, beta, gamma = abg[0], abg[1], abg[2]
        theta = self._theta 
        
        dx, dt = self._dx, self._dt;
        xs, ts = self._xs, self._ts;

        #Allocate memory for solution:
        Nus = zeros((self._num_phis(),
                     self._num_steps(),
                      self._num_nodes() ));
        
        #Impose TCs: (already done by construction
#        Nus[:, -1, :] = zeros_like(self._num_nodes())
        #Impose left BCs: (already done by construction):
#        Nus[:, :, 0] = zeros_like(self._num_nodes())
       
        #Diffusion coefficient:
        D = -beta * beta / 2.; #the diffusion coeff
        
        #Solve it using C-N/C-D:
        
        #Allocate MASS MTX:    
        M = lil_matrix((self._num_nodes() - 1, self._num_nodes() - 1));

        #Time and Reaction terms:
        e = (-1.- .5*dt) * ones(self._num_nodes() - 1); 
        
        #Diffusion term:
        dx_sqrd = dx * dx;
        d_on = D * dt / dx_sqrd;
        d_off = -.5 * D * dt / dx_sqrd;
        centre_diag = e + d_on;
        M.setdiag(centre_diag)
        
        #TIME MARCH:
        for tk in xrange(self._num_steps()-1, 0, -1):
            t_next = ts[tk]
            t_prev = ts[tk-1]
            #iterate over the phases:
            for phi, phi_idx in zip(self._phis, xrange(0, self._num_phis() )):
                #Rip the next time solution (already solved):
                Nu_next = Nus[phi_idx, tk, :];
                 
                #Advection coefficient:
                U_next = -(alpha - xs + gamma * sin(theta * (t_next + phi)));
                U_prev = -(alpha - xs + gamma * sin(theta * (t_prev + phi)));
        
                #Form the Right-hand side:
                Nu_extrapolated = Nu_next[-1] + (U_next[-1] * Nu_next[-1] - Ls[phi_idx, tk]) / D * dx
                L_next =    .5 * Nu_next[1:] + \
                            U_next[1:] * r_[ ((Nu_next[2:] - Nu_next[:-2]) / (2.* dx),           
                                               (Nu_next[-1] - Nu_next[-2]) / dx)] + \
                            D * r_[(diff(Nu_next, 2),
                                    Nu_extrapolated -2*Nu_next[-1] + Nu_next[-2] )] / dx_sqrd; #diffusion term - Nu_extrapolated term is derived by extrapolating \nu using the Robin  BCs
                RHS = -Nu_next[1:] + .5 * dt * L_next;
                
                #Incorporate time-dependent terms (advection) into the 'mass' matrix:
                u_off = .5 * dt / (2.*dx) * U_prev[1:];
                
                L_left = d_off + u_off;
                M.setdiag(r_[(L_left[1:-1], -1.0)], -1);
                
                L_right = d_off - u_off;
                M.setdiag(r_[(L_right[:-1])], 1);
                
                #Impose the right (Robin) BCs:
                RHS[-1] = (-Ls[phi_idx, tk-1]);
                M[-1,-2] = D / dx;
                M[-1,-1] = -D / dx - U_prev[-1];
                
                #Thomas Solve it:
                Mx = M.tocsr()
                Nu_prev = spsolve(Mx, RHS);
                if visualize:
                    if rand() < 1./ (1+ log(self._num_steps())):
                        figure()
                        plot(xs[1:], Nu_prev); title('t=' + str(t_next) + ' \phi = %.2g'%phi);
                    
                #Store solution:
                Nus[phi_idx, tk-1, 1:] = Nu_prev;  

        return Nus
    
    
    def solveAdjointUpwinded(self, abg, Ls, visualize=False):
        ''' abg are the params, 
        Ls is the deviation from the data (Fs|_\vt - S )(t)  of the solution of the forward problem,
        ...
        We are solving BACKWARDS!!!'''
        alpha, beta, gamma = abg[0], abg[1], abg[2]
        theta = self._theta 
        
        dx, dt = self._dx, self._dt;
        xs, ts = self._xs, self._ts;

        #Allocate memory for solution:
        Nus = zeros((self._num_phis(),
                     self._num_steps(),
                      self._num_nodes() ));
        
        #Impose TCs: (already done by construction
#        Nus[:, -1, :] = zeros_like(self._num_nodes())
        #Impose left BCs: (already done by construction):
#        Nus[:, :, 0] = zeros_like(self._num_nodes())
       
        #Diffusion coefficient:
        D = -beta * beta / 2.; #the diffusion coeff
        
        #Solve it using C-N/C-D:
        
        #Allocate MASS MTX:    
        M = lil_matrix((self._num_nodes() - 1, self._num_nodes() - 1));

        #Time and Reaction terms:
        e = (-1.- .5*dt) * ones(self._num_nodes() - 1); 
        
        #Diffusion term:
        dx_sqrd = dx * dx;
        d_on = D * dt / dx_sqrd;
        d_off = -.5 * D * dt / dx_sqrd;
#        centre_diag = e + d_on;
#        M.setdiag(centre_diag)
        
        #TIME MARCH:
        for tk in xrange(self._num_steps()-1, 0, -1):
            t_next = ts[tk]
            t_prev = ts[tk-1]
            #iterate over the phases:
            for phi, phi_idx in zip(self._phis, xrange(0, self._num_phis() )):
                #Rip the next time solution (already solved):
                Nu_next = Nus[phi_idx, tk, :];
                 
                #Advection coefficient:
                U_next = -(alpha - xs + gamma * sin(theta * (t_next + phi)));
                U_prev = -(alpha - xs + gamma * sin(theta * (t_prev + phi)));
        
                #Form the Right-hand side:
                Nu_extrapolated = Nu_next[-1] + (U_next[-1] * Nu_next[-1] - Ls[phi_idx, tk]) / D * dx
                L_next =    .5 * Nu_next[1:] + \
                            U_next[1:] * r_[ ((Nu_next[2:] - Nu_next[:-2]) / (2.* dx),           
                                               (Nu_next[-1] - Nu_next[-2]) / dx)] + \
                            D * r_[(diff(Nu_next, 2),
                                    Nu_extrapolated -2*Nu_next[-1] + Nu_next[-2] )] / dx_sqrd; #diffusion term - Nu_extrapolated term is derived by extrapolating \nu using the Robin  BCs
                RHS = -Nu_next[1:] + .5 * dt * L_next;
                
                #Incorporate upwinded time-dependent terms (advection) into the 'mass' matrix:
                u_left  = -ones_like(U_prev[0:-1]) *.5 * dt / dx * U_prev[0:-1]
                u_right = ones_like(U_prev[1:]) *.5 * dt/ dx * U_prev[1:]
                u_on    = ones_like(U_prev[1:]) *.5 * dt/ dx * U_prev[1:]
                
                #zero out the irrelevant entries
                u_left [U_prev[:-1] >=0] = .0
                u_on   [U_prev[1:] >=0] *= -1.
                u_right[U_prev[1:] <=0] = .0
                
                L_left = d_off + u_left;
                M.setdiag(r_[(L_left[1:-1], -1.0)], -1);
                
                L_right = d_off - u_right;
                M.setdiag(r_[(L_right[:-1])], 1);
                
                centre_diag = e + d_on + u_on;
                M.setdiag(centre_diag)
        
                #Impose the right (Robin) BCs:
                RHS[-1] = (-Ls[phi_idx, tk-1]);
                M[-1,-2] = D / dx;
                M[-1,-1] = -D / dx - U_prev[-1];
                
                #Thomas Solve it:
                Mx = M.tocsr()
                Nu_prev = spsolve(Mx, RHS);
                if visualize:
                    if rand() < 1./ (1+ log(self._num_steps())):
                        figure()
                        plot(xs[1:], Nu_prev); title('t=' + str(t_next) + ' \phi = %.2g'%phi);
                    
                #Store solution:
                Nus[phi_idx, tk-1, 1:] = Nu_prev;  

        return Nus
    
    def _getICs(self, xs, dx):
        initial_distribution = ones_like(xs) * (xs > .0);
        initial_distribution[where(abs(xs) < dx * 1e-4)] = .5;
        
        return array(list(initial_distribution)*self._num_phis()).reshape((self._num_phis(),-1)); 
    
    def outerPlus(self, a, b):
        return (a.ravel()[:,newaxis]+b.ravel()[newaxis,:]).flatten()
    
    def MOLsolve(self, abg, visualize=False):
        alpha, beta, gamma = abg[0], abg[1], abg[2];  theta = self._theta 
        
        dx = self._dx;
        xs, ts = self._xs, self._ts;

        #Impose ICs:
        F0 = self._getICs(xs[1:], dx).flatten()
#        F0 = np.rollaxis(F0,1)
        N_F = (self._num_nodes() - 1)*self._num_phis() 
        
        
        D = beta * beta / 2.; #the diffusion coeff
        #Needed for Newmann bC imposition:
        last_indxs = (self._num_nodes() - 1)*arange(1,self._num_phis()+1) - 1;
        #Allocate the evolution mtx:
        A = lil_matrix((N_F, N_F));
        
        dx_sqrd = dx * dx;
        d_on = -2 * D / dx_sqrd * ones(N_F);
        d_off = D / dx_sqrd;

        centre_diag =  d_on
        A.setdiag(centre_diag)

        def rhs(t,F):
            U =  (alpha + self.outerPlus(gamma * sin(theta * (t + self._phis)), -xs[1:]));
            u_off = U / (2*dx);           
        
            left_diag   =  d_off + u_off
            right_diag  =  d_off - u_off
            
            #impose the right BCs:
            A.setdiag(left_diag[1:]  , -1);
            A.setdiag(right_diag[:-1], 1);

            #Evolve it:
            Mx = A.tocsr()
            dF  = Mx*F;
            dF[last_indxs] = dF[last_indxs-1]; 

            return dF; 

        #Assimulo Problem+Solver:
        exp_mod = Explicit_Problem(rhs, F0)
        exp_sim = CVode(exp_mod) #Create a CVode solver
    
        #Sets parameters:
        exp_sim.iter  = 'Newton' #Default 'FixedPoint'
        exp_sim.discr = 'Adams' #Default 'Adams'
        exp_sim.maxord = 3;        
        exp_sim.atol = [1e-3] #Default 1e-6
#        exp_sim.maxh = self._dt
        exp_sim.rtol = 1e-3 #Default 1e-6
        
        #Simulate:
        ts, Fs = exp_sim.simulate(self._ts[-1], len(self._ts[1:]))

        #Return:
#        Fs = Fs.reshape( (Fs.shape[0], len(self._xs[1:]) , -1) )
        NX = len(self._xs[1:])
        lF = ones((self._num_phis(), Fs.shape[0], NX));
        for phi_idx in xrange(self._num_phis()):
            lF[phi_idx, :, :] = Fs[:, phi_idx*NX: (phi_idx+1)*NX]
        return lF
    
    def estimateParameterGradient(self, abg, Fs, Nus):
        beta = abg[1];
                
        dx, dt = self._dx, self._dt;
        dx_sqrd = dx*dx;
        
        theta = self._theta
        ts = outer(self._ts[1:], ones_like(self._xs[1:]))
        
        dGdp = zeros((len(abg), self._num_phis()));
                
        for phi, phi_idx in zip(self._phis, xrange(0, self._num_phis() )):
            lF = Fs[phi_idx,1:,:]
            
            lNu = Nus[phi_idx,1:, 1:]
            
            Fx = (lF[:, 1:] - lF[:,0:-1]) / dx;
            #using the Newmann BCs here:
            minus_betaFxx = -beta*(c_[(lF[:, 2:] - 2*lF[:,1:-1] + lF[:,:-2],
                                       -lF[:, -1] + lF[:,-2])])/ dx_sqrd
            
            sinFx = sin(theta*(ts+phi))*Fx;
            
            dGdp[0, phi_idx] = sum(lNu*Fx)           *dx*dt
            dGdp[1, phi_idx] = sum(lNu*minus_betaFxx)*dx*dt
            dGdp[2, phi_idx] = sum(lNu*sinFx)        *dx*dt
           
        return sum(dGdp, 1)
    
    def transformSurvivorData(self, binnedTrain):
        Ss = empty((self._num_phis(),
                     self._num_steps()))
        bins = binnedTrain.bins;
        for phi, phi_idx in zip(self._phis, xrange(0, self._num_phis() )):
           
#        for phi, phi_idx in zip(bins.keys(), xrange(len(bins.keys()))):
            
            unique_Is = bins[phi]['unique_Is']
            SDF = bins[phi]['SDF']
            
            Ss [phi_idx, :] = interp(self._ts, r_[(.0,unique_Is)],
                                               r_[(1., SDF)])
        
        return Ss;
                
    
def SolverDriver():
    dx = .0025; dt = FPMultiPhiSolver.calculate_dt(dx, 1.)
    phis =  linspace(.05, .95, 4)
    theta = 1.0
    S = FPMultiPhiSolver(theta, phis,
                         dx, dt,
                         Tf=.5)
        
    start =  time.clock()
    Fs = S.solve((.1, 1.2, 2.), visualize=False)
    print 'solve time = ', time.clock() - start;

    figure()
    for phi, phi_idx in zip(S._phis, xrange(0, S._num_phis())) :
        subplot(ceil(S._num_phis() / 2.), 2, phi_idx+1)
        plot(S._ts, Fs[phi_idx, :, -1], 'g');
        title('$F_b$ , $\phi = %.2g$'%phi);
        ylim((.0,1.05))
    
        
    
#def AssimuloSolveDriver():
#    dx = .025; dt = FPMultiPhiSolver.calculate_dt(dx, 2.)
#    phis =  linspace(.05, .95, 4)
#    theta = 1.0
#    S = FPMultiPhiSolver(theta, phis,
#                         dx, dt,
#                         Tf=2.5)
#    
#    abg = (1., .5, 1.)
#    start =  time.clock()
#    Fs = S.MOLsolve(abg, visualize=False)
#    print 'MOL time = ', time.clock() - start;
#
#    figure()
#    for t_idx in [0,-1]:
#        subplot(2,1,abs(t_idx)+1)
#        plot(S._xs[1:], Fs[0, t_idx, :]);
#        title('$F(x)$');
#
#    figure(1)
#    for phi, phi_idx in zip(S._phis, xrange(0, S._num_phis())):
#        subplot(ceil(S._num_phis() / 2.), 2, phi_idx+1)
#        plot(S._ts, Fs[phi_idx, :, -1]);
#        title('$F_b$ , $\phi = %.2g$'%phi);
#        ylim((.0,1.05))
#        
#        
#    start =  time.clock()
#    Fs = S.solve(abg, visualize=False)
#    print 'CN time = ', time.clock() - start;
#
#    figure(1)
#    hold(True)
#    for phi, phi_idx in zip(S._phis, xrange(0, S._num_phis())) :
#        subplot(ceil(S._num_phis() / 2.), 2, phi_idx+1)
#        plot(S._ts, Fs[phi_idx, :, -1], 'r');
#        title('$F_b$ , $\phi = %.2g$'%phi);
#        ylim((.0,1.05))


def ConvergenceDriver():
    dx = .05; dt = FPMultiPhiSolver.calculate_dt(dx, 2.)
    phis =  linspace(.05, .95, 4)
    theta = 1.0; Tf= 2.5
    abg = (.1, .2, 3.)

    S = FPMultiPhiSolver(theta, phis,
                         dx, dt,
                         Tf)
    
    Fs = S.solve(abg, visualize=False)
    
    figure(1)
    hold (True)
    for phi, phi_idx in zip(S._phis, xrange(0, S._num_phis())) :
        subplot(ceil(S._num_phis() / 2.), 2, phi_idx+1)
        plot(S._xs, Fs[phi_idx, -1, :], 'r');
        title('$F(x) - $\phi = %.2g$'%phi);
        ylim((.0,1.05))

    figure(2)
    hold(True)
    for phi, phi_idx in zip(S._phis, xrange(0, S._num_phis())) :
        subplot(ceil(S._num_phis() / 2.), 2, phi_idx+1)
        plot(S._ts, Fs[phi_idx, :, -1] ,'r');
        title('$F_b$ , $\phi = %.2g$'%phi);
        ylim((.0,1.05))
        
    
    ##################################
    dx = .001; dt = FPMultiPhiSolver.calculate_dt(dx, 1.)
    S = FPMultiPhiSolver(theta, phis,
                         dx, dt,
                         Tf)
    Fs = S.solve(abg, visualize=False)
    
    figure(1)
    hold (True)
    for phi, phi_idx in zip(S._phis, xrange(0, S._num_phis())) :
        subplot(ceil(S._num_phis() / 2.), 2, phi_idx+1)
        plot(S._xs, Fs[phi_idx, -1, :], 'g');
        ylim((.0,1.05))

    figure(2)
    hold(True)
    for phi, phi_idx in zip(S._phis, xrange(0, S._num_phis())) :
        subplot(ceil(S._num_phis() / 2.), 2, phi_idx+1)
        plot(S._ts, Fs[phi_idx, :, -1]);
        ylim((.0,1.05))
        
def trailAverage(Ls, N=5):
    xs = Ls[-1::-1]
    
    M = len(xs)
    block = empty((M, N))
    for n in xrange(N):
        block[:, n] = r_[(n*[xs[0]], xs[:M-n])]
    
    ys = sum(block, 1) / N

    return ys[-1::-1]
     
def AdjointSolverDriver():
        N_phi = 20;
        print 'N_phi = ', N_phi
        
        phis =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)
        
#        file_name = 'sinusoidal_spike_train_N=1000_superT_13'
#        file_name = 'sinusoidal_spike_train_N=1000_subT_2'
        file_name = 'sinusoidal_spike_train_N=1000_crit_5'
    
        intervalStats(file_name)
        binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phis)

        phi_omit = None
    #    phi_omit = r_[(linspace(.15, .45, 4),
    #                   linspace(.55,.95, 5) )]  *2*pi/ binnedTrain.theta
    
        binnedTrain.pruneBins(phi_omit, N_thresh = 100, T_thresh= 16.)
        print 'N_bins = ', len(binnedTrain.bins.keys())
        
        Tf = binnedTrain.getTf() #/ 2.
        print 'Tf = ', Tf
    
        params = binnedTrain._Train._params
        da, db, dg = .5, .05, .25;
        abg = (params._alpha+da, params._beta+db, params._gamma+dg)
        abg = (0.717, 0.206, 0.522)
        print abg
        phis = binnedTrain.bins.keys();
        theta = binnedTrain.theta
        xmin = FPMultiPhiSolver.calculate_xmin(Tf, abg, theta)
        max_speed = abg[0] + abs(abg[2]) - xmin;
        dx = abg[1] / max_speed / 2e1; #dx = .025;
        dt = FPMultiPhiSolver.calculate_dt(dx, abg, xmin, factor = 8.)
        print 'xmin = ', xmin
        print 'dx, dt = ', dx, dt
        S = FPMultiPhiSolver(theta, phis,
                         dx, dt, Tf, xmin)

        start =  time.clock()
        Fs = S.solve(abg, visualize=False)
        Ss = S.transformSurvivorData(binnedTrain)
        Ls = Fs[:,:,-1] - Ss; 
#        Nus = S.solveAdjoint(abg, Ls)

        L2s = empty_like(Ls);
        lsq_order = 4;
        for phi_idx in xrange(S._num_phis()):
#            L2s[phi_idx,:] = smooth(Ls[phi_idx, :])
#            L2s[phi_idx,:] = polyval(polyfit(S._ts, Ls[phi_idx, :], lsq_order),S._ts)
            L2s[phi_idx,:] = trailAverage(Ls[phi_idx, :], 6)
           
        Nus = S.solveAdjointUpwinded(abg, Ls)
        Nu2s = S.solveAdjointUpwinded(abg, L2s)
        
        dGdp = S.estimateParameterGradient(abg, Fs, Nu2s)
        print 'dGdp = ', dGdp
        print 'solve+adjoint time = ', time.clock() - start;
        
        fig_side = figure()
        
        mpl.rcParams['figure.subplot.hspace'] = .25
        N_cols = 4;
        for phi, phi_idx in zip(S._phis, xrange(0, S._num_phis())) :
            fig_side.add_subplot(S._num_phis(), N_cols, N_cols*phi_idx+1)
            plot(S._ts, Fs[phi_idx, :, -1], 'g', label='F');
            plot(S._ts, Ss[phi_idx, :], 'r', label='G');
            title('$F_{th}, G$ , $\phi = %.2g$'%phi);
            ylim((.0,1.05))
            legend(loc='lower left')
        
            subplot(S._num_phis(), N_cols, N_cols*phi_idx+2)
            plot(S._ts, Ls[phi_idx, :], 'r', label='raw');
            plot(S._ts, L2s[phi_idx, :], 'k', label='Smoothed');
            title('$F_{th} - G$');
#            ylim((-.1,.1))     
#            ylim((-1.025,1.025))
            legend()
            
            subplot(S._num_phis(), N_cols, N_cols*phi_idx+3)
            plot(S._ts, Nus[phi_idx, :, -1], 'k');
            title('$\\nu  : dx=%.3g, dt=%.3g$'%(dx, dt)  , fontsize=16);
#            ylim((-1e3,1e3)) 
            
            subplot(S._num_phis(), N_cols, N_cols*phi_idx+4)
            plot(S._ts, Nu2s[phi_idx, :, -1], 'k');
            title(r'$\nu_{upwind} : dx=%.3g, dt=%.3g$'%(dx, dt) , fontsize=16);
#            ylim((-1e1,1e1))
        get_current_fig_manager().window.showMaximized()
        
        ##SURFACE PLOTS:
#        from mpl_toolkits.mplot3d import Axes3D
#        fig = figure();
#        for phi, phi_idx in zip(S._phis, xrange(0, S._num_phis())) :
#            ax = fig.add_subplot(S._num_phis(), 1, phi_idx+1, projection='3d')
#            X, Y = np.meshgrid(S._xs, S._ts)
#            ax.plot_surface(X, Y, Nus[phi_idx,:,:], rstride=2, cstride=2, cmap=cm.jet,
#                                linewidth=0, antialiased=False)
#            xlabel('x'); ylabel('t')
#            title('$\\nu  - \phi = %.2g : dx=%.3g, dt=%.3g$'%(phi, dx, dt) );
#            
#        fig = figure();
#        for phi, phi_idx in zip(S._phis, xrange(0, S._num_phis())) :
#            ax = fig.add_subplot(S._num_phis(), 1, phi_idx+1, projection='3d')
#            X, Y = np.meshgrid(S._xs, S._ts)
#            ax.plot_surface(X, Y, Nu2s[phi_idx,:,:], rstride=2, cstride=2, cmap=cm.jet,
#                                linewidth=0, antialiased=False)
#            xlabel('x'); ylabel('t')
#            title('$\\nu2  - \phi = %.2g : dx=%.3g, dt=%.3g$'%(phi, dx, dt) );

#def smooth(x,window_len=11,window='hanning'):
#    """smooth the data using a window with requested size.
#    
#    This method is based on the convolution of a scaled window with the signal.
#    The signal is prepared by introducing reflected copies of the signal 
#    (with the window size) in both ends so that transient parts are minimized
#    in the begining and end part of the output signal.
#    
#    input:
#        x: the input signal 
#        window_len: the dimension of the smoothing window; should be an odd integer
#        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
#            flat window will produce a moving average smoothing.
#
#    output:
#        the smoothed signal
#        
#    example:
#
#    t=linspace(-2,2,0.1)
#    x=sin(t)+randn(len(t))*0.1
#    y=smooth(x)
#    
#    see also: 
#    
#    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
#    scipy.signal.lfilter
# 
#    TODO: the window parameter could be the window itself if an array instead of a string   
#    """
#    from numpy import  hanning, hamming, bartlett, blackman
#
#    if x.ndim != 1:
#        raise ValueError, "smooth only accepts 1 dimension arrays."
#
#    if x.size < window_len:
#        raise ValueError, "Input vector needs to be bigger than window size."
#
#
#    if window_len<3:
#        return x
#
#
#    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
#        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
#
#
#    s=r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
#    #print(len(s))
#    if window == 'flat': #moving average
#        w=ones(window_len,'d')
#    else:
#        w=eval(window+'(window_len)')
#
#    y= convolve(w/w.sum(),s,mode='valid')
#    return y

def AdjointTestSinglePhi():
    theta = 2.
    phis = array([.1])*2.*pi/theta    
    abg = [.5, .3,.5]
    print abg
    x_min = -.5
                
    dx = .025; dt = FPMultiPhiSolver.calculate_dt(dx, abg[0]+abg[2]-(x_min), 1.0)
    Tf = 5.0
        
    S = FPMultiPhiSolver(theta, phis,
                         dx, dt,
                         Tf, X_MIN = x_min)
    start =  time.clock()
    Fs = S.solve(abg, visualize=False)
    
    Ls = .25*linspace(.0, 1.0, len(S._ts)).reshape((1,len(S._ts)));
    Nus = S.solveAdjoint(abg, Ls)
    Nus_upwind = S.solveAdjointUpwinded(abg, Ls)
    print 'solve time = ', time.clock() - start
    
    fig_side = figure()
    
    from mpl_toolkits.mplot3d import Axes3D
    N_cols = 3;
    for phi, phi_idx in zip(S._phis, xrange(0, S._num_phis())) :
        fig_side.add_subplot(S._num_phis(), N_cols, N_cols*phi_idx+1)
        plot(S._ts, Fs[phi_idx, :, -1], 'g');
        title('$F_{th}(t)$ , $\phi = %.2g$'%phi);
        ylim((.0,1.05))
    
        subplot(S._num_phis(), N_cols, N_cols*phi_idx+2)
        plot(S._ts, Ls[phi_idx, :], 'r');
        title('$\Lambda(t)$');
#        ylim((-1.,1.))     
        
        subplot(S._num_phis(), N_cols, N_cols*phi_idx+3); hold(True)
        plot(S._ts, Nus[phi_idx, :, -1], 'k', label='CD');
        plot(S._ts, Nus_upwind[phi_idx, :, -1], 'b', label='Upwind');
        title('$\\nu  : dx=%.3g, dt=%.3g$'%(dx, dt)  , fontsize=16);
        ylim((-1e1,1e1)) 
        legend()
        
    get_current_fig_manager().window.showMaximized()
    
    
def AdjointTestDriver():
    N_phi = 20;
    print 'N_phi = ', N_phi
    
    phis =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)
    
    file_name = 'sinusoidal_spike_train_N=1000_superT_13'
#        file_name = 'sinusoidal_spike_train_N=1000_subT_2'
#    file_name = 'sinusoidal_spike_train_N=1000_crit_5'

#    intervalStats(file_name)
    binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phis)

    phi_omit = None
#    phi_omit = r_[(linspace(.15, .45, 4),
#                   linspace(.55,.95, 5) )]  *2*pi/ binnedTrain.theta
    binnedTrain.pruneBins(phi_omit, N_thresh = 100, T_thresh= 10.)
    print 'N_bins = ', len(binnedTrain.bins.keys())
    
    Tf = binnedTrain.getTf() #/ 2.
    print 'Tf = ', Tf

    params = binnedTrain._Train._params
    abg_true = (params._alpha, params._beta, params._gamma)
    print 'true = ',     abg_true

    phis = binnedTrain.bins.keys();
    theta = binnedTrain.theta
           
    for da in [-.5, .75]:
        for db in [-.1, .25]:
            for dg in [-.25, .5]:
                abg = (params._alpha+da, params._beta+db, params._gamma+dg)
                
                xmin = FPMultiPhiSolver.calculate_xmin(Tf, abg)
                max_speed = abg[0] + abs(abg[2]) - xmin;
                dx = abg[1] / max_speed / 1e1;   #dx = .025;
                dt = FPMultiPhiSolver.calculate_dt(dx, abg, xmin, factor = 8.)
                S = FPMultiPhiSolver(theta, phis,
                                 dx, dt, Tf, xmin)

                Fs = S.solve(abg, visualize=False)
                Ss = S.transformSurvivorData(binnedTrain)
                Ls = Fs[:,:,-1] - Ss
                Nus = S.solveAdjoint(abg, Ls)
            
                dGdp = S.estimateParameterGradient(abg, Fs, Nus)
                print '#'*64
                print 'CASE::(%.2g, %.2g, %.2g)'%(abg[0],abg[1],abg[2])
                print 'G = ', .5*sum(Ls*Ls)*S._dt 
                print 'dGdp = ', dGdp
                print 'Expected sign :', sign(array(abg) - array(abg_true))
                print 'Computed sign :', sign(dGdp)
        
        
def AdjointManualEstimator():
    N_phi = 20;
    print 'N_phi = ', N_phi
    
    phis =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)
    
#    file_name = 'sinusoidal_spike_train_N=1000_superT_13'
#    file_name = 'sinusoidal_spike_train_N=1000_subT_2'
    file_name = 'sinusoidal_spike_train_N=1000_crit_5'

#    intervalStats(file_name)
    binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phis)

    phi_omit = None
#    phi_omit = r_[(linspace(.15, .45, 4),
#                   linspace(.55,.95, 5) )]  *2*pi/ binnedTrain.theta
    binnedTrain.pruneBins(phi_omit, N_thresh = 100, T_thresh= 16.)
    print 'N_bins = ', len(binnedTrain.bins.keys())
    
    Tf = binnedTrain.getTf() #/ 2.
    print 'Tf = ', Tf

    params = binnedTrain._Train._params
    abg_true = (params._alpha, params._beta, params._gamma)
    print 'true = ',     abg_true

    phis = binnedTrain.bins.keys();
    theta = binnedTrain.theta
    da = 1.; db = .5; dg = .75
    abg = (params._alpha+da, params._beta+db, params._gamma+dg)

    abg = initialize_right_2std(binnedTrain)
    
    start = time.clock()
    for n in xrange(16):
        xmin = FPMultiPhiSolver.calculate_xmin(Tf, abg)
        max_speed = abg[0] + abs(abg[2]) - xmin;
        dx = abg[1] / max_speed / 1e1;   #dx = .025;
        dt = FPMultiPhiSolver.calculate_dt(dx, abg, xmin, factor = 8.)
        S = FPMultiPhiSolver(theta, phis,
                         dx, dt, Tf, xmin)

        Fs = S.solve(abg, visualize=False)
        Ss = S.transformSurvivorData(binnedTrain)
        Ls = Fs[:,:,-1] - Ss
        Nus = S.solveAdjoint(abg, Ls)
    
        dGdp = S.estimateParameterGradient(abg, Fs, Nus)

        from numpy.linalg.linalg import norm
        
        G = .5*sum(Ls*Ls)*S._dt 
        factor = G / Tf  

        abg = abg - factor*dGdp/ norm(dGdp) 
        
        print n
        print 'abg = (%.3g, %.3g, %.3g)'%(abg[0],abg[1],abg[2])
        print 'G = ', G 
        print 'dGdp = ', dGdp
       
    print 'time = ', time.clock() - start
    
    
def AdjointTrueParamsTester():
    phis =  linspace(.05, .95, 10)
    phi_omit = None
    
    for file_name in ['sinusoidal_spike_train_T=20000_subT_13.path']:
#                      'sinusoidal_spike_train_T=5000_crit_14.path',
#                      'sinusoidal_spike_train_T=5000_superSin_8.path',
#                      'sinusoidal_spike_train_T=5000_superT_16.path'  ]:
        print file_name
        
        binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phis)
        binnedTrain.pruneBins(phi_omit, N_thresh = 96, T_thresh = 8.)
        print 'N_bins = ', len(binnedTrain.bins.keys())
    
        Tf = binnedTrain.getTf()
        print 'Tf = ', Tf
       
        solver_phis = binnedTrain.bins.keys();
        theta = binnedTrain.theta
        x_min = -.5;
        
        ps = binnedTrain._Train._params
        abg_true = array((ps._alpha, ps._beta, ps._gamma))
        print 'abg_true = ', abg_true
        abg_Abg =  [.5, .25, .3]; #abs( abg_true + .2*abg_true*randn(3) );
        abg_aBg =  [.3, .5, .2]; #abs( abg_true + .2*abg_true*randn(3) );
        abg_abG =  [.25, .2, .5]; #abs( abg_true + .2*abg_true*randn(3) );
        
        for abg, test_case in zip([abg_true, abg_Abg, abg_aBg, abg_abG],
                                  ['exact', 'Abg', 'aBg', 'abG']):
            max_speed = abg[0] + abs(abg[2]) - x_min;
            dx = abg[1] / max_speed / 1e2
            dt = FPMultiPhiSolver.calculate_dt(dx, max_speed, .5e2)
            
            S = FPMultiPhiSolver(theta, solver_phis,
                                 dx, dt,
                                 Tf, X_MIN = x_min)
            Fs = S.solve(abg, visualize=False)
            Ss = S.transformSurvivorData(binnedTrain)
            Ls = Fs[:,:,-1] - Ss; #    Ls = zeros_like(Ss)
#            L2s = empty_like(Ls);
#            lsq_order = 6;
#            for phi_idx in xrange(S._num_phis()):
#                L2s[phi_idx,:] = polyval(polyfit(S._ts, Ls[phi_idx, :], lsq_order) ,
#                                         S._ts)
            Nus = S.solveAdjointUpwinded(abg, Ls)
            
            dGdp = S.estimateParameterGradient(abg, Fs, Nus)
            print test_case, ':(%.2g, %.2g, %.2g)'%(abg[0],abg[1],abg[2])
            print 'G = ', .5*sum(Ls*Ls)*S._dt 
            print 'dGdp = ', dGdp
#            print '||dGdp||_2 = ', sqrt(sum(dGdp*dGdp))
            print 'Sign =', sign(dGdp)
            print 'Expected Sign =', sign(abg - abg_true)

        print '#'*64        

def add_inner_title(ax, title, loc, size=None, **kwargs):
            from matplotlib.offsetbox import AnchoredText
            from matplotlib.patheffects import withStroke
            if size is None:
                size = dict(size=plt.rcParams['legend.fontsize'])
            at = AnchoredText(title, loc=loc, prop=size,
                              pad=0., borderpad=0.5,
                              frameon=False, **kwargs)
            ax.add_artist(at)
            at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
            return at
    
def visualizeData_vs_FP(S, abgt, binnedTrain, title_tag = '', save_fig_name = ''):
        #Visualize time:
        abg = abgt
        
        mpl.rcParams['figure.subplot.left'] = .15
        mpl.rcParams['figure.subplot.right'] = .95
        mpl.rcParams['figure.subplot.bottom'] = .1
        mpl.rcParams['figure.subplot.top'] = .9
        
        figure()
        bins = binnedTrain.bins
        phis = bins.keys()
        Tf = binnedTrain.getTf()
        S.setTf(Tf)
        
        Fs = None
        if 3 == len(abgt):
            Fs = S.solve(abg)
        elif 4 == len(abgt):
            Fs = S.solve_tau(abgt)
            
        ts = S._ts;

        for (phi_m, err_idx) in zip( sort(phis),
                                     xrange(len(phis)) ):
            phi_idx = nonzero( abs(phis - phi_m) < 1e-4)
            lF = squeeze(Fs[phi_idx, :,-1])
            ax = subplot(len(phis),1, err_idx + 1)
            hold (True)
            plot(ts, lF, 'b',linewidth=3, label='Analytic'); 
            plot(bins[phi_m]['unique_Is'], 
                 bins[phi_m]['SDF'], 'r+', markersize = 10, label='Data')
            if (0 == err_idx):
                if '' != title_tag:
                     title(title_tag + ' $(\\alpha, \\beta, \gamma) = (%.2g,%.2g,%.2g)$' %(abg[0], abg[1], abg[2]), fontsize = 36)
                ylabel(r'$\bar{G}(t)$', fontsize = 32)
                legend()
            if len(bins.keys()) != err_idx+1:
                setp(ax.get_xticklabels(), visible=False)
            else:
                xlabel('$t$', fontsize = 32)
                for label in ax.xaxis.get_majorticklabels():
                    label.set_fontsize(24)
                for label in ax.yaxis.get_majorticklabels():
                    label.set_fontsize(24)   

            ylim((.0, 1.05))
               
#            annotate('$\phi_{norm}$ = %.2g'%(phi_m / 2 / pi * binnedTrain.theta), (.1, .5), fontsize = 24 )
        
        if '' != save_fig_name:
            file_name = os.path.join(FIGS_DIR, save_fig_name+'.png')
            print 'saving to ', file_name
            savefig(file_name) 

def visualizeFP(S, abgt, theta, Tf,
                title_tag = '', save_fig_name = '', phis= [.0]):
        #Visualize time:
        abg = abgt
        print abg;
        
        mpl.rcParams['figure.subplot.left'] = .15
        mpl.rcParams['figure.subplot.right'] = .95
        mpl.rcParams['figure.subplot.bottom'] = .1
        mpl.rcParams['figure.subplot.top'] = .9
        
        S.setTf(Tf)
        
        Fs = None
        if 3 == len(abgt):
            Fs = S.c_solve(abg)
        elif 4 == len(abgt):
            Fs = S.solve_tau(abgt)
            
        ts = S._ts;
        print ts[0],ts[-1]

        #SDF Fig:
        figure()
        def relabel_major(x, pos):
            if x < 0:
                    return ''
            else:
                    return '$%.1f$' %x
        for (phi_m, err_idx) in zip( sort(phis),
                                     xrange(len(phis)) ):
            phi_idx = nonzero( abs(phis - phi_m) < 1e-4)
            lF = squeeze(Fs[phi_idx, :,-1])
            ax = subplot(len(phis),1, err_idx + 1)
            hold (True)
            plot(ts, lF, 'b',linewidth=3, label='Analytic'); 
            annotate(r'$ \phi_{m} =  %.2g \cdot 2\pi $' %(phi_m / 2 / pi * theta),
                      (Tf/(1.5), .6), fontsize = 24 )
            ylabel(r'$\bar{G}(t)$', fontsize = 24)
            xlim((.0, Tf))
            if (0 == err_idx):
                if '' != title_tag:
#                    title(title_tag + ' $(\\alpha, \\beta, \gamma) = (%.2g,%.2g,%.2g)$' %(abg[0], abg[1], abg[2]), fontsize = 36)
                    title(title_tag, fontsize = 36)
                
            if len(phis) != err_idx+1:
                setp(ax.get_xticklabels(), visible=False)
            else:
                xlabel('$t$', fontsize = 20)
                locs, labels = xticks()
                locs = locs[1:]
                xticks(locs)
                ax.xaxis.set_major_formatter(FuncFormatter(relabel_major))
                for label in ax.xaxis.get_majorticklabels():
                    label.set_fontsize(20)
            yticks( [0., .5,  1.0] )
            ax.yaxis.set_major_formatter(FuncFormatter(relabel_major))
            for label in ax.yaxis.get_majorticklabels():
                label.set_fontsize(20)   

            ylim((.0, 1.05))
        
        get_current_fig_manager().window.showMaximized()
        if '' != save_fig_name:
            file_name = os.path.join(FIGS_DIR, save_fig_name+'.png')
            print 'saving to ', file_name
            savefig(file_name) 
        #pdf Fig:
        dt = S._dt;
        figure()

        for (phi_m, err_idx) in zip( sort(phis),
                                     xrange(len(phis)) ):
            phi_idx = nonzero( abs(phis - phi_m) < 1e-4)
            lF = squeeze(Fs[phi_idx, :,-1])
            
            lg = -diff(lF) / dt
            ymax = 1.05*amax(lg) 
            
            ax = subplot(len(phis),1, err_idx + 1)
            hold (True)
            plot(ts[:-1], lg, 'b',linewidth=3, label='Analytic'); 
            annotate(r'$ \phi_{m} =  %.2g \cdot 2\pi $' %(phi_m / 2.0 / pi * theta),
                      (2*Tf/3., ymax * .6), fontsize = 24)
            ylabel('$g(t)$', fontsize = 24)
            if (0 == err_idx):
                if '' != title_tag:
#                    title(title_tag + ' $(\\alpha, \\beta, \gamma) = (%.2g,%.2g,%.2g)$' %(abg[0], abg[1], abg[2]), fontsize = 36)
                    title(title_tag, fontsize = 36)
                
#                legend()
            if len(phis) != err_idx+1:
                setp(ax.get_xticklabels(), visible=False)
            else:
                xlabel('$t$', fontsize = 24)
                locs, labels = xticks()
                locs = locs[1:]
                xticks(locs)
                ax.xaxis.set_major_formatter(FuncFormatter(relabel_major)) 
                for label in ax.xaxis.get_majorticklabels():
                    label.set_fontsize(20)
#            ax.yaxis.set_major_locator(MaxNLocator(3))    
            max_lg = amax(lg) 
            yticks([.0, max_lg / 2., max_lg])

            ax.yaxis.set_major_formatter(FuncFormatter(relabel_major))    
            for label in ax.yaxis.get_majorticklabels():
                label.set_fontsize(20) 
                
            ylim((.0, ymax))
            xlim((.0, Tf))
        get_current_fig_manager().window.showMaximized()
        if '' != save_fig_name:
            file_name = os.path.join(FIGS_DIR, save_fig_name+'_pdf.png')
            print 'saving to ', file_name
            savefig(file_name)
        
    
    
def VisualizeSinusoidallyDominating():
        N_phi = 20;
        print 'N_phi = ', N_phi
        
        phis =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)
        phi_omit = None
        dx = .025; dt = FPMultiPhiSolver.calculate_dt(dx, 5., 2.)
    
        file_name = 'sinusoidal_spike_train_N=1000_superSin_11'
        
        binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phis)
        binnedTrain.pruneBins(phi_omit, N_thresh = 64)
        print 'N_bins = ', len(binnedTrain.bins.keys())
    
        Tf = binnedTrain.getTf()
        print 'Tf = ', Tf
       
        solver_phis = binnedTrain.bins.keys();
        theta = binnedTrain.theta;
       
        S = FPMultiPhiSolver(theta, solver_phis,
                             dx, dt, Tf, X_MIN=-4.0)
    
        ps = binnedTrain._Train._params
        abg_true = array((ps._alpha, ps._beta, ps._gamma))

        abg = abg_true
        visualizeData_vs_FP(S, abg, binnedTrain, 'TRUE: 4.0',
                            save_fig_name='SuperSin_true_params')
        get_current_fig_manager().window.showMaximized()
#        abg_est =  [ 0.52476609  ,0.46863966,  0.52519313]
#        abg = abg_est
#        visualizeData_vs_FP(S, abg, binnedTrain, '$N_{\\phi} = %d$ '%N_phi)
#        get_current_fig_manager().window.showMaximized()
       
#        abg_est =  [.494,.140,1.11]
#        abg = abg_est
#        visualizeData_vs_FP(S, abg, binnedTrain, 'F-P: ',
#                            save_fig_name='SuperSin_NM_est')
#        get_current_fig_manager().window.showMaximized()
#       
#       
#        abg_est =  [.541,.181, .983]
#        abg = abg_est
#        visualizeData_vs_FP(S, abg, binnedTrain, 'Fortet: ',
#                            save_fig_name='SuperSin_Fortet_est')
#        get_current_fig_manager().window.showMaximized()
    

def VisualizeEffectOfM(save_figs=False):
#    file_name = 'sinusoidal_spike_train_T=20000_subT_11.path'
    file_name = 'sinusoidal_spike_train_N=1000_superSin_11'
    
#        mpl.rcParams['figure.subplot.left'] = .05
#        mpl.rcParams['figure.subplot.right'] = .95
    mpl.rcParams['figure.subplot.bottom'] = .125
#        mpl.rcParams['figure.subplot.top'] = .9
    mpl.rcParams['figure.subplot.wspace'] = .2
#        mpl.rcParams['figure.subplot.hspace'] = .55
    mpl.rcParams['figure.figsize'] = 15.5, 4.5
 
    inner_titles = {0: 'A',
                    1:'B',
                    2:'C',
                    3:'D'}
    figure()
    label_font_size = 16
    for M_idx,N_phi in enumerate([5, 10, 20, 40]):
        print 'N_phi = ', N_phi
        
        phis =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)
        binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phis)
        Tf = binnedTrain.getTf()
        print 'Sample Tf = ', Tf
        
        N_thresh = max([len(bin['Is']) for bin in binnedTrain.bins.values() ])
        
        print 'Num samples in max bin = ', N_thresh
        
        binnedTrain.pruneBins(None, N_thresh = N_thresh, T_thresh=16.)
#        print 'N_bins = ', len(binnedTrain.bins.keys())
    
        Tf = binnedTrain.getTf()
        print 'Tf in largest bin = ', Tf
       
        solver_phis = binnedTrain.bins.keys();
        theta = binnedTrain.theta;

        ps = binnedTrain._Train._params
        abg_true = array((ps._alpha, ps._beta, ps._gamma))

        abg = abg_true

        dx = .025; 
        x_min = FPMultiPhiSolver.calculate_xmin(Tf, abg, theta)
        dt = FPMultiPhiSolver.calculate_dt(dx,abg, x_min)
      
        S = FPMultiPhiSolver(theta, solver_phis,
                             dx, dt, Tf, x_min)
        bins = binnedTrain.bins
        phis = bins.keys()
        Tf = binnedTrain.getTf()
        S.setTf(Tf)
        
        Fs = S.solve(abg)
        ts = S._ts;

        ax = subplot(1,4, M_idx + 1)
        phi_m =  phis[0]
        phi_idx = nonzero( abs(phis - phi_m) < 1e-4)
        lF = squeeze(Fs[phi_idx, :,-1])
        hold (True)
        plot(ts, lF, 'b',linewidth=2, label='Analytic'); 
        plot(bins[phi_m]['unique_Is'], 
             bins[phi_m]['SDF'], 'r+', markersize = 6, label='Data')
        legend()
        if (0 == M_idx):
            ylabel(r'$\bar{G}(t)$', fontsize = label_font_size)
        else:
            setp(ax.get_yticklabels(), visible=False)
        xlabel('$t$', fontsize = label_font_size)
        for label in ax.xaxis.get_majorticklabels():
            label.set_fontsize(label_font_size)
        for label in ax.yaxis.get_majorticklabels():
            label.set_fontsize(label_font_size)   

        ylim((.0, 1.05))
        t = add_inner_title(ax, inner_titles[M_idx], loc=3, size=dict(size=16))
        t.patch.set_ec("none")
        t.patch.set_alpha(0.5)
        
#    get_current_fig_manager().window.showMaximized() 
    if True == save_figs:
        file_name = os.path.join(FIGS_DIR, 'EffectOfM.pdf')
        print 'saving to ', file_name
        savefig(file_name, dpi=(300)) 

    
def VisualizeEffectOfM_RefereeReply(save_figs=False,
                                    N_thresh = 5):
#    file_name = 'sinusoidal_spike_train_T=20000_subT_11.path'
    file_name = 'sinusoidal_spike_train_N=1000_superSin_11'
    
    mpl.rcParams['figure.subplot.left'] = .175
#        mpl.rcParams['figure.subplot.right'] = .95
    mpl.rcParams['figure.subplot.bottom'] = .125
#        mpl.rcParams['figure.subplot.top'] = .9
    mpl.rcParams['figure.subplot.wspace'] = .2
#        mpl.rcParams['figure.subplot.hspace'] = .55
    mpl.rcParams['figure.figsize'] = 15.5, 9
 
    inner_titles = {0:{0: 'A',
                        1:'B',
                        2:'C',
                        3:'D'},
                    1: {0:'E',
                        1:'F',
                        2:'G',
                        3:'I'}}
    figure()
    label_font_size = 16
    xlabel_font_size = 32
    for M_idx,N_phi in enumerate([5, 10, 20, 40]):
        print 'N_phi = ', N_phi
        
        phis =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)
        binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phis)
        
        bins = binnedTrain.bins
        Nms = []
        for key in bins.keys():
            Is = bins[key]['Is']
#            print '%.2f'%key, len(Is), '%.2f'%amax(Is) 
            Nms.append( len(Is))
        print Nms
        Nms = array(Nms)
        Nms = Nms[Nms>N_thresh]
        Nms = [amin(Nms), amax(Nms)]
        print Nms
        
        for key in bins.keys():
            Is = bins[key]['Is']
            Nm = len(Is);
            if Nm not in Nms:
                del bins[key]
#        print len(binnedTrain.bins)
            
        for N_idx, key in enumerate(bins.keys()):
            Is = bins[key]['Is']
            Tf = amax(Is)
            Nm = len(Is)

       
            solver_phis = [key];
            theta = binnedTrain.theta;

            ps = binnedTrain._Train._params
            abg_true = array((ps._alpha, ps._beta, ps._gamma))

            abg = abg_true

            dx = .02; 
            x_min = FPMultiPhiSolver.calculate_xmin(Tf, abg, theta)
            dt = FPMultiPhiSolver.calculate_dt(dx,abg, x_min)
      
            S = FPMultiPhiSolver(theta, solver_phis,
                                 dx, dt, Tf, x_min)
#        bins = binnedTrain.bins
#        phis = bins.keys()

            S.setTf(Tf)
#            //Solve it:        
            Fs = S.solve(abg)
            ts = S._ts;

            ax = subplot(2,4, N_idx *4 + (M_idx + 1))
            phi_m =  phis[0]

            lF = squeeze(Fs[0, :,-1])
            hold (True)
            plot(ts, lF, 'b',linewidth=2, label='Analytic'); 
            plot(bins[key]['unique_Is'], 
                    bins[key]['SDF'], 'r+', markersize = 6, label='Data')
            if(1 ==N_idx and 40 == N_phi):
                legend()
            
            annotate(r'$ \phi_{m} =  %.2g \cdot 2\pi $' %(key / 2 / pi * theta),
                      (.1, .35), fontsize = 20 )
            annotate(r'$ N_m =  %.d$' %(Nm),
                      (.1, .23), fontsize = 20 )
            
            if (0 == M_idx):
                ylabel(r'$\bar{G}_{\phi}(t)$', fontsize = xlabel_font_size)
                row_label = ''
                if (0 == N_idx):
                    row_label = 'Least populous\n bin'
                else:
                    row_label = 'Most populous\n bin'
                        
                text(-.8, 0.5, row_label,
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform = ax.transAxes,
                        rotation='vertical',
                        size = xlabel_font_size - 4)
                    
            else:  #M_idx > 0
                setp(ax.get_yticklabels(), visible=False)
            if (0 == N_idx):
                title('$M = %d$'%N_phi, fontsize = xlabel_font_size)
            if (1 == N_idx):
                xlabel('$t$', fontsize = xlabel_font_size)
            for label in ax.xaxis.get_majorticklabels():
                label.set_fontsize(label_font_size)
            for label in ax.yaxis.get_majorticklabels():
                label.set_fontsize(label_font_size)
            if (0 == N_idx):
                xlim(.0, amax([ts[-1], 1.]))
            

            ylim((.0, 1.05))
            t = add_inner_title(ax, inner_titles[N_idx][M_idx], loc=3, 
                                size=dict(size=ABCD_LABEL_SIZE))
            t.patch.set_ec("none")
            t.patch.set_alpha(0.5)
        
#    get_current_fig_manager().window.showMaximized() 
    if True == save_figs:
        file_name = os.path.join(FIGS_DIR, 'EffectOfM_Referees.pdf')
        print 'saving to ', file_name
        savefig(file_name, dpi=(300)) 

    
def VisualizeCritical():
        N_phi = 20;
        print 'N_phi = ', N_phi
        
        phis =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)
        phi_omit = None
        dx = .025; dt = FPMultiPhiSolver.calculate_dt(dx, 5., 2.)
    
        file_name = 'sinusoidal_spike_train_T=10000_crit.path'
        
        binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phis)
        binnedTrain.pruneBins(phi_omit, N_thresh = 200, T_thresh=10.)
        print 'N_bins = ', len(binnedTrain.bins.keys())
    
        Tf = binnedTrain.getTf()
        print 'Tf = ', Tf
       
        solver_phis = binnedTrain.bins.keys();
        theta = binnedTrain.theta;
    
        S = FPMultiPhiSolver(theta, solver_phis,
                             dx, dt, Tf)
    
        ps = binnedTrain._Train._params
        abg_true = array((ps._alpha, ps._beta, ps._gamma))

        abg = abg_true
        visualizeData_vs_FP(S, abg, binnedTrain, '',
                            save_fig_name='Critical_true_params')
        get_current_fig_manager().window.showMaximized()
       
        abg_est =  [0.804, 0.38 , 0.498]
        
        abg = abg_est
        visualizeData_vs_FP(S, abg, binnedTrain, '',
                            save_fig_name='Critical_init_params')
       

  
def VisualizeSubThresh():
        N_phi = 20;
        print 'N_phi = ', N_phi
        
        phis =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)
        phi_omit = None
        dx = .04; dt = FPMultiPhiSolver.calculate_dt(dx, 4., 2.)
    
        file_name = 'sinusoidal_spike_train_T=20000_subT_13.path'
        
        binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phis)
        binnedTrain.pruneBins(phi_omit, N_thresh = 64, T_thresh=32.)
        print 'N_bins = ', len(binnedTrain.bins.keys())
    
        Tf = binnedTrain.getTf()
        print 'Tf = ', Tf
       
        solver_phis = binnedTrain.bins.keys();
        theta = binnedTrain.theta;
    
        ps = binnedTrain._Train._params
        abg_true = array((ps._alpha, ps._beta, ps._gamma))
    
        S = FPMultiPhiSolver(theta, solver_phis,
                             dx, dt, Tf, X_MIN = -.5)
        abg = abg_true
        visualizeData_vs_FP(S, abg, binnedTrain,  'X_MIN = -.5',
                            save_fig_name='')
        get_current_fig_manager().window.showMaximized()
        

        S = FPMultiPhiSolver(theta, solver_phis,
                             dx, dt, Tf, X_MIN = -2.)
        abg = abg_true
        visualizeData_vs_FP(S, abg, binnedTrain,  'X_MIN = -2',
                            save_fig_name='')
        get_current_fig_manager().window.showMaximized()
        
#        abg_est =  [.44, .07, -1.019]
#        abg = abg_est
#        visualizeData_vs_FP(S, abg, binnedTrain, '',
#                            save_fig_name='')
       

def visualizeExampleNoData(abg = [1.5, .3, 1.0],
                           save_fig_name = '',
                           theta = 2.0,
                           phi = 1.0):
    Tf = 15.0
    dx = .0125
    x_min = -.5
    
    S = FPMultiPhiSolver(theta, [phi],
                         dx, FPMultiPhiSolver.calculate_dt(dx, abg,  x_min, 2.0), Tf,  x_min)
    
    Fs = S.solve(abg)
            
    ts = S._ts;
    xs = S._xs

    #Visualize time:
    mpl.rcParams['figure.subplot.left'] = .05
    mpl.rcParams['figure.subplot.right'] = .975
    mpl.rcParams['figure.subplot.bottom'] = .05
    mpl.rcParams['figure.subplot.top'] = .925

    for az in [-50, -10]:  #arange(-60, 0, 5):
        fig = figure();
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.view_init(elev = None, azim= az)
        X, Y = np.meshgrid(xs, ts)
        ax.plot_surface(X, Y, Fs[0,:,:], rstride=2, cstride=2, cmap=cm.jet,
                                    linewidth=0, antialiased=False)
        xlabel('x', fontsize = 18); ylabel('t',fontsize = 24)
#        title('$F, az = %d$'%az, fontsize = 36)
        title('$F$', fontsize = 36)
#        get_current_fig_manager().window.showMaximized()
        
        if '' != save_fig_name:
                file_name = os.path.join(FIGS_DIR, 'surf_' +save_fig_name+'_az='+str(az) +'.png')
                print 'saving to ', file_name
                savefig(file_name) 
    
    mpl.rcParams['figure.subplot.left'] = .125
    mpl.rcParams['figure.subplot.bottom'] = .175
    figure()
    lF = squeeze(Fs[0, :,-1])
    ax = subplot(111)
    plot(ts, lF, 'b', linewidth = 8); 
    xlabel('$t$', fontsize = 24)
    ylabel('$F_{th}$', fontsize = 24)
    
    for label in ax.xaxis.get_majorticklabels():
        label.set_fontsize(20)
    for label in ax.yaxis.get_majorticklabels():
        label.set_fontsize(20)
    
    ylim((.0, 1.05))
    
#    get_current_fig_manager().window.showMaximized()
    
    if '' != save_fig_name:
            file_name = os.path.join(FIGS_DIR, 'SDF_' +save_fig_name+'.png')
            print 'saving to ', file_name
            savefig(file_name) 
    
    
def visualizeSolverIn3D(abg = [1.5, .3, 1.0],
                           save_fig = True,
                           theta = 1.0,
                           phi = 1.0):
    Tf = 15.0
    dx = .025

    x_min = -.5
    
    S = FPMultiPhiSolver(theta, [phi],
                         dx, FPMultiPhiSolver.calculate_dt(dx, abg,  x_min, 2.0), Tf,  x_min)
    
    Fs = S.solve(abg)
            
    ts = S._ts;
    xs = S._xs

    #Visualize time:
    mpl.rcParams['figure.figsize'] = 17, 10
    mpl.rcParams['figure.subplot.left'] = .05
    mpl.rcParams['figure.subplot.right'] = .975
    mpl.rcParams['figure.subplot.bottom'] = .1
    mpl.rcParams['figure.subplot.top'] = .925
    mpl.rcParams['figure.dpi'] = 300

    inner_titles = {0: 'A',
                   1:'B',
                   2:'C',
                   3:'D'}
    xlabel_font_size = 32
    label_font_size = 20
    
    fig = figure();
    for az_idx, az in enumerate([-60, -15, 40]):  #arange(-60, 0, 5):
#    for az_idx, az in enumerate([-60]):
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(2, 2, 1+az_idx, projection='3d')
        ax.view_init(elev = None, azim= az)
        X, Y = np.meshgrid(xs, ts)
        ax.plot_surface(X, Y, Fs[0,:,:], rstride=2, cstride=2, cmap=cm.jet,
                                    linewidth=0, antialiased=False)
        xlabel(r'$x$', fontsize = xlabel_font_size);
        ylabel(r'$t$',fontsize = xlabel_font_size)
#        ax.set_zlabel(r'$F$', fontsize = xlabel_font_size)
        ticks = [-.5, .0, .5, 1]
        ax.set_xticks(ticks)
        ax.set_xticklabels([r'$%.1f$'%tick for tick in ticks])
#        xticks(ticks, [r'$%.1f$'%tick for tick in ticks])
        ticks = [0, 3, 12, 15]
        ax.set_yticks(ticks)
        ax.set_yticklabels([r'$%d$'%tick for tick in ticks])
#        yticks(ticks, [r'$%d$'%tick for tick in ticks])
        ticks = [.0, .5,  1.]
        ax.set_zticks(ticks)
        ax.set_zticklabels([r'$%.1f$'%tick for tick in ticks])
        
#        ticks, labels = yticks()
#        yticks(ticks[::2])
#        ticks, labels = xticks()
#        xticks(ticks[::2])
#        title('$F, az = %d$'%az, fontsize = 36)
#        title('$F$', fontsize = 36)

        t = add_inner_title(ax, inner_titles[az_idx], loc=3,
                             size=dict(size=ABCD_LABEL_SIZE))
        t.patch.set_ec("none")
        t.patch.set_alpha(0.5)
        for label in ax.xaxis.get_majorticklabels():
            label.set_fontsize(label_font_size)
        for label in ax.yaxis.get_majorticklabels():
            label.set_fontsize(label_font_size  )
        for label in ax.zaxis.get_majorticklabels():
            print label
            label.set_label('$1$')
#            label.set_text('$1$')
            label.set_fontsize(label_font_size)
    
    
    
    lF = squeeze(Fs[0, :,-1])
    ax = subplot(2,2,4)
    plot(ts, lF, 'b', linewidth = 6); 
    xlabel('$t$', fontsize = xlabel_font_size)
    ylabel('$F_{th}$', fontsize = xlabel_font_size)
    
    for label in ax.xaxis.get_majorticklabels():
        label.set_fontsize(label_font_size )
    for label in ax.yaxis.get_majorticklabels():
        label.set_fontsize(label_font_size )
    t = add_inner_title(ax, inner_titles[3], loc=3,
                         size=dict(size=ABCD_LABEL_SIZE))
    t.patch.set_ec("none")
    t.patch.set_alpha(0.5)
    
    ylim((.0, 1.05))
    
    get_current_fig_manager().window.showMaximized()
    if '' != save_fig:
            file_name = os.path.join(FIGS_DIR, 'SDF3D_combined.pdf') 
            print 'saving to ', file_name
            savefig(file_name) 
            
#def VisualizeFourRegimes():    
#        N_phi = 20;
#        print 'N_phi = ', N_phi
#        
#        phis =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)
#        phi_omit = None
#           
#        for regime_name, N_thresh, T_thresh in zip(['superT', 'crit','subT',  'superSin'],
#                                                   array([90, 97, 118, 96]),
#                                                   [4., 16., 32, 16.]):
#
#            regime_label = 'sinusoidal_spike_train_N=1000_' + regime_name + '_12' 
#        
#            binnedTrain = BinnedSpikeTrain.initFromFile(regime_label, phis)
#            binnedTrain.pruneBins(phi_omit, N_thresh = N_thresh, T_thresh=T_thresh)
#            print 'N_bins = ', len(binnedTrain.bins.keys())
#        
#            Tf = binnedTrain.getTf()
#            print 'Tf = ', Tf
#           
#            solver_phis = binnedTrain.bins.keys();
#            theta = binnedTrain.theta;
#    
#            ps = binnedTrain._Train._params
#            abg_true = array((ps._alpha, ps._beta, ps._gamma))
#    
#            abg = abg_true
#
#            xmin = FPMultiPhiSolver.calculate_xmin(Tf, abg)
#            dx = .025; dt = FPMultiPhiSolver.calculate_dt(dx, abg, xmin, factor = 5.)
#            print 'xmin = ', xmin
#            print 'dx, dt = ', dx, dt
#            S = FPMultiPhiSolver(theta, solver_phis,
#                             dx, dt, Tf, xmin)
#
#            visualizeData_vs_FP(S, abg, binnedTrain, regime_name + ' : ',
#                                save_fig_name='Illustrate_refined'+regime_name)
#            get_current_fig_manager().window.showMaximized()
    

def Visualize_ThetaBeta():
    for beta in [.1, .25, .5, 1, 2.0 ]:
        for theta in [10]:
#    for beta in [.1 ]:
#        for theta in [10.0]:
            params = [.5, beta, (.5)*sqrt(1. + theta**2), theta]
                     
            Tf = 3*2*pi / theta ;
            print 'Tf = ', Tf
            print 'beta = ', beta, 'theta = ', theta
    
            abg = params[0:3]
            xmin = FPMultiPhiSolver.calculate_xmin(Tf, abg, theta)
            dx = .0125; dt = FPMultiPhiSolver.calculate_dt(dx, abg, xmin, factor = 1.)
            print 'xmin = ', xmin, ', dx, dt = ', dx, dt
            solver_phis =  2*pi/theta*array([0,.25, .5, .75])
            S = FPMultiPhiSolver(theta, solver_phis,
                                 dx, dt, Tf, xmin)
            visualizeFP(S, abg, theta, Tf,
                            title_tag = r'$\beta = %.1f, \theta= %.1f:\alpha=%.1f\gamma=%.2f$'%(beta, theta, params[0], params[2]),
#                            save_fig_name='critical_vary_betatheta_b%d_th%d'%(beta*10, theta*10), phis =solver_phis)
                            phis =solver_phis)
#    for tag in ['', '_pdf']:
#        for theta in [.2, 1, 2.0, 5.0, 10.0]:
#            for beta in [.1, 1, 5.]:
#                print r'\subfloat[b =%.2f]'%beta
#                print r'{\includegraphics[width =0.33\textwidth,height=.25\textheight]{Figs/FP/'+ \
#                        'superSin_vary_betatheta_b%d_th%d'%(beta*10, theta*10) +\
#                        tag + '.png}}'
#            print r'\\'
#        print ''
            
    

def VisualizeFourRegimes(save_figs = True):    
        N_phi = 4;
        print 'N_phi = ', N_phi
        
        phis =  2*pi*array([0,.25, .5, .75])
        regime_tags = {'superT':'Supra-Threshold', 'crit':'Critical',
                       'subT':'Sub-Threshold',  'superSin':'Super-Sinusoidal'}
       
        mpl.rcParams['figure.figsize'] = 17, 5*4
        mpl.rcParams['figure.subplot.left'] = .15
        mpl.rcParams['figure.subplot.right'] = .95
        mpl.rcParams['figure.subplot.top'] = .975
        mpl.rcParams['figure.subplot.bottom'] = .05
        mpl.rcParams['figure.subplot.hspace'] = .375
        mpl.rcParams['figure.subplot.wspace'] = .3
        
        pdf_fig = figure()
        sdf_fig = figure()
        inner_titles = {0: 'A',
                       1:'B',
                       2:'C',
                       3:'D'}
        
        def add_inner_title(ax, title, loc, size=None, **kwargs):
            from matplotlib.offsetbox import AnchoredText
            from matplotlib.patheffects import withStroke
            if size is None:
                size = dict(size=plt.rcParams['legend.fontsize'])
            at = AnchoredText(title, loc=loc, prop=size,
                              pad=0., borderpad=0.5,
                              frameon=False, **kwargs)
            ax.add_artist(at)
            at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
            return at
        def relabel_major(x, pos):
                if x < 0:
                        return ''
                else:
                        return '$%.1f$' %x
        
        for regime_idx, regime_name in enumerate(['superT', 'superSin', 'crit','subT']):
#        for regime_name in ['superT']:
            regime_label = 'sinusoidal_spike_train_N=1000_' + regime_name + '_12' 
            binnedTrain = BinnedSpikeTrain.initFromFile(regime_label, phis)
            Tf = binnedTrain.getTf()
            print 'Tf = ', Tf
            
            theta = binnedTrain.theta;
            print 'theta = ', theta
            ps = binnedTrain._Train._params
            abg_true = array((ps._alpha, ps._beta, ps._gamma))
            abg = abg_true
            xmin = FPMultiPhiSolver.calculate_xmin(Tf, abg, theta)
            dx = .0125; 
            dt = FPMultiPhiSolver.calculate_dt(dx, abg, xmin, factor = 5.)
            print 'xmin = ', xmin, ', dx, dt = ', dx, dt
            
            S = FPMultiPhiSolver(theta, phis,
                             dx, dt, Tf, xmin)

#            visualizeData_vs_FP(S, abg, binnedTrain, regime_name + ' : ',
#                                save_fig_name='Illustrate_refined'+regime_name)
#            visualizeFP(S, abg,theta, Tf,  regime_tags[regime_name],
#                    save_fig_name='Illustrate4'+regime_name, phis =solver_phis)

            S.setTf(Tf)
            Fs = S.c_solve(abg)
            ts = S._ts;
        
            #SDF Fig:
            figure(sdf_fig.number)
            label_font_size = 24
            xlabel_font_size = 40
            
            for (phi_m, err_idx) in zip( sort(phis),
                                         xrange(len(phis)) ):
                phi_idx = nonzero( abs(phis - phi_m) < 1e-4)
                lF = squeeze(Fs[phi_idx, :,-1])
                ax = sdf_fig.add_subplot(2*len(phis),2, 8*(regime_idx>1) +mod(regime_idx,2)  + 2*err_idx + 1)
                plot(ts, lF, 'b',
                     linewidth=3, label='Analytic'); 
                annotate(r'$ \phi_{m} =  %.2g \cdot 2\pi $' %(phi_m / 2 / pi),
                          (Tf/(1.75), .5), fontsize = label_font_size )
                ylabel(r'$\bar{G}(t)$', fontsize = xlabel_font_size)
                xlim((.0, Tf))

                if len(phis) != err_idx+1:
                    setp(ax.get_xticklabels(), visible=False)
                else:
                    if regime_idx > 1:
                        xlabel('$t$', fontsize = xlabel_font_size)
                    locs, labels = xticks()
                    locs = locs[1:]
                    xticks(locs)
                    ax.xaxis.set_major_formatter(FuncFormatter(relabel_major))
                    for label in ax.xaxis.get_majorticklabels():
                        label.set_fontsize(label_font_size)
                yticks( [0., .5,  1.0] )
                ax.xaxis.set_major_locator(MaxNLocator(6)) 
                ax.yaxis.set_major_formatter(FuncFormatter(relabel_major))
                for label in ax.yaxis.get_majorticklabels():
                    label.set_fontsize(label_font_size)   
                t = add_inner_title(ax, inner_titles[regime_idx], loc=3,
                                     size=dict(size=ABCD_LABEL_SIZE))
                t.patch.set_ec("none")
                t.patch.set_alpha(0.5)
        
                ylim((.0, 1.05))
 
            #pdf Fig:
            dt = S._dt;
            figure(pdf_fig.number)
            for (phi_m, err_idx) in zip( sort(phis),
                                         xrange(len(phis)) ):
                phi_idx = nonzero( abs(phis - phi_m) < 1e-4)
                lF = squeeze(Fs[phi_idx, :,-1])
                
                lg = -diff(lF) / dt
                ymax = 1.05*amax(lg) 
                
                ax = pdf_fig.add_subplot(2*len(phis),2, 8*(regime_idx>1) +mod(regime_idx,2)  + 2*err_idx + 1)
                ax.plot(ts[:-1], lg, 'b',linewidth=2, label='Analytic'); 
                ax.annotate(r'$ \phi_{m} =  %.2g \cdot 2\pi $' %(phi_m / 2.0 / pi * theta),
                          (2*Tf/3., ymax * .5), fontsize = label_font_size)
                ylabel('$g(t)$', fontsize = xlabel_font_size)
                if len(phis) != err_idx+1:
                    setp(ax.get_xticklabels(), visible=False)
                else:
                    if regime_idx > 1:
                        xlabel('$t$', fontsize = xlabel_font_size)
                    locs, labels = xticks()
                    locs = locs[1:]
                    xticks(locs)
                    ax.xaxis.set_major_formatter(FuncFormatter(relabel_major)) 
                    for label in ax.xaxis.get_majorticklabels():
                        label.set_fontsize(label_font_size)
                max_lg = amax(lg) 
                yticks([.0, max_lg / 2., max_lg])
                ax.xaxis.set_major_locator(MaxNLocator(6)) 
                t = add_inner_title(ax, inner_titles[regime_idx], loc=3, 
                                    size=dict(size=ABCD_LABEL_SIZE))
                t.patch.set_ec("none")
                t.patch.set_alpha(0.5)
                
                ax.yaxis.set_major_formatter(FuncFormatter(relabel_major))    
                for label in ax.yaxis.get_majorticklabels():
                    label.set_fontsize(label_font_size) 
                    
                ylim((.0, ymax))
                xlim((.0, Tf))

        
#        tight_layout()
    
        if save_figs:
            file_name = os.path.join(FIGS_DIR, 'regimes_sdf_combined.pdf')
            print 'saving to ', file_name
            sdf_fig.savefig(file_name, dpi=(300))
            file_name = os.path.join(FIGS_DIR, 'regimes_pdf_combined.pdf')
            print 'saving to ', file_name
            pdf_fig.savefig(file_name, dpi=(300))
       

def visualizeTauEstimates(N_thresh = 90, T_thresh = 4.0):
#    file_name = 'sinusoidal_spike_train_N=1000_superT_9'
    file_name = 'sinusoidal_spike_train_N=1000_crit_14'
    print file_name
    
    N_phi = 20;
    print 'N_phi = ', N_phi
    
    phi_norms =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)
    
    binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
    binnedTrain.pruneBins(None, N_thresh, T_thresh)
    print 'N_bins = ', len(binnedTrain.bins.keys())
    
    Tf = binnedTrain.getTf()
    print 'Tf = ', Tf 

    solver_phis = binnedTrain.bins.keys();
    theta = binnedTrain.theta;
    
    ps = binnedTrain._Train._params
    abg_true = array((ps._alpha, ps._beta, ps._gamma))

    abg = abg_true
    dx = .02;
    xmin = FPMultiPhiSolver.calculate_xmin(Tf, abg)
    dt = FPMultiPhiSolver.calculate_dt(dx, abg, xmin)
    S = FPMultiPhiSolver(theta, solver_phis,
                         dx, dt, Tf, X_min=xmin)
    visualizeData_vs_FP(S, abg, binnedTrain, r'True, $\tau = 1.$',
                        save_fig_name='Tau_true')
    get_current_fig_manager().window.showMaximized()
    
#    abg_est =  [ 1.87489463  ,0.31415139  ,0.91759959  ,0.61929902]
    abg_est =  [ 0.94657025  ,0.52626892,  0.55184594,  0.53410389]
    abg = abg_est
    visualizeData_vs_FP(S, abg, binnedTrain, r'$\tau = %.2g$'%abg[3], 
                        save_fig_name='Tau_est')
    get_current_fig_manager().window.showMaximized()
    
#    for tau in [.25, .5, 1.25, 2.5]:
#        abg = r_[(abg_true[:3]  ,tau)]
#        
#        visualizeData_vs_FP(S, abg, binnedTrain, r'$\tau = %.2g$'%tau,
#                            save_fig_name='Tau_wayoff_%.2g'%tau)
#        get_current_fig_manager().window.showMaximized()


def CvsPYsolver(abg = [1.5, .3, 1.0],
                   save_fig_name = ''):    
    
    abg = [.3, .5, .3]
    Tf = 10.0;
    dx = .025;
    x_min = FPMultiPhiSolver.calculate_xmin(Tf, abg);
    print 'Xmin = ', x_min;
        
    theta = 2.0
    N_phi = 4;
    phis =  (linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)) * 2*pi/theta
    
    S = FPMultiPhiSolver(theta, phis,
                         dx, FPMultiPhiSolver.calculate_dt(dx, abg, x_min), Tf, x_min)
    
    #TODO: time this call!!!
    start =  time.clock()
    Pys = S.solve(abg)
    print 'Py time = ', time.clock() - start;
    
    start =  time.clock()
    Cs = S.c_solve(abg)
    print 'C time = ', time.clock() - start;
            
    print 'max error = ', max(Cs-Pys)
    ts = S._ts;
    xs = S._xs

    thresh_fig = figure();
    for Fs, solver_lang, solver_graph in zip([Pys, Cs],
                                             ["PYTHON", "C/C++"],
                                             ['g+', 'r*']):
        #Visualize time:
        mpl.rcParams['figure.subplot.left'] = .05
        mpl.rcParams['figure.subplot.right'] = .975
        mpl.rcParams['figure.subplot.bottom'] = .05
        mpl.rcParams['figure.subplot.top'] = .925
        
        az = -20 #in arange(-60, -5, 5):
        fig = figure();
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.view_init(elev = None, azim= az)
        X, Y = np.meshgrid(xs, ts)
        ax.plot_surface(X, Y, Fs[0,:,:], rstride=2, cstride=2, cmap=cm.jet,
                                    linewidth=0, antialiased=False)
        xlabel('x', fontsize = 18); ylabel('t',fontsize = 24)
        #        title('$F, az = %d$'%az, fontsize = 36)
        title(solver_lang + '::$F(x,t)$ ', fontsize = 36)
#        title("Python")
        get_current_fig_manager().window.showMaximized()
        
        #    if '' != save_fig_name:
        #            file_name = os.path.join(FIGS_DIR, 'surf_' +save_fig_name+'_az='+str(az) +'.png')
        #            print 'saving to ', file_name
        #            savefig(file_name) 
        
        mpl.rcParams['figure.subplot.left'] = .125
        mpl.rcParams['figure.subplot.bottom'] = .175
#        figure()
        for phi_idx in xrange(N_phi):
            lF = squeeze(Fs[0, :,-1])
            
            ax = thresh_fig.add_subplot(N_phi, 1, phi_idx+1)            
            ax.plot(ts, lF, solver_graph, linewidth = 8, label=solver_lang); 
            ax.set_xlabel('$t$', fontsize = 24)
            ax.set_ylabel('$F_{th}$', fontsize = 24)
            
            for label in ax.xaxis.get_majorticklabels():
                label.set_fontsize(20)
            for label in ax.yaxis.get_majorticklabels():
                label.set_fontsize(20)
            ax.legend()
                        
            ylim((.0, 1.05))
            
            
        get_current_fig_manager().window.showMaximized()
        
        if '' != save_fig_name:
                file_name = os.path.join(FIGS_DIR, 'SDF_' +save_fig_name+'.png')
                print 'saving to ', file_name
                savefig(file_name) 
    
    
def testGradPhiF(abg = [.5,     .3, .5* sqrt(1. + 1.0**2)],
                 theta = 1.0, Tf = 4.0):
    
    xmin = FPMultiPhiSolver.calculate_xmin(Tf, abg, theta)
    dx = .02; 
    dt = FPMultiPhiSolver.calculate_dt(dx, abg, xmin, factor = 1.)
    print 'xmin = ', xmin, ', dx, dt = ', dx, dt
    solver_phis =  2*pi/theta*array([0,.25, .5, .75])
    S = FPMultiPhiSolver(theta, solver_phis,
                         dx, dt, Tf, xmin)
    
    Fs = S.c_solve(abg)
    Fphis = S.solveFphi(abg, Fs)
    ts = S._ts
    for phi_idx, phi in enumerate(S._phis):
        
        SDF     = Fs[phi_idx,:,-1]
        SDF_phi = Fphis[phi_idx,:,-1]
        
        pdf = -diff(SDF) / dt;
        pdf_phi = -diff(SDF_phi) / dt;
        
        figure()
        subplot(221)
        plot(ts, SDF);
        title('SDF');
        subplot(223);
        plot(ts[:-1], pdf);
        title('pdf');
        
        subplot(222)
        plot(ts, SDF_phi);
        title('SDF_phi');
        subplot(224);
        plot(ts[:-1], pdf_phi);
        title('pdf_phi');

                
if __name__ == '__main__':
    from pylab import *
#    mpl.rcParams['figure.dpi'] = 300
#    visualizeExampleNoData(abg = [.5,     .3, .5* sqrt(1. + 1.0**2)],
#                           save_fig_name='phi_110', theta = 1.0, phi = 1.10)    
#    visualizeExampleNoData(abg = [.5,     .3, .5* sqrt(1. + 1.0**2)],
#                           save_fig_name='phi_100', theta = 1.0, phi = 1.0)

#    ConvergenceDriver()
#    SolverDriver()
#    AssimuloSolveDriver()

#    AdjointSolverDriver()
#    AdjointTestDriver()
#    AdjointManualEstimator()
    
#    AdjointTrueParamsTester()
#    AdjointManualEstimators()

#    VisualizeSinusoidallyDominating()
#    VisualizeSubThresh()
#    VisualizeCritical()
#    Visualize_ThetaBeta()
#    VisualizeFourRegimes()
    
#    visualizeTauEstimates()

    visualizeSolverIn3D(abg = [.5, .3, .5* sqrt(1. + 1.0**2)],
                           theta = 1.0, phi = pi / 2.0)
    
#    VisualizeEffectOfM()
#    VisualizeEffectOfM_RefereeReply(True)    


#    CvsPYsolver()   
    
    
#    theta = 16.0
#    testGradPhiF(abg = [.3,     .3, .5* sqrt(1. + theta**2)],
#                 theta = theta, Tf = 16.0);
    
    show()
    