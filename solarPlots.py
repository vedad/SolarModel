import numpy as np
import matplotlib.pyplot as plt

def read():
	
	global R_sun; R_sun = 6.96e10
	global M0, R0, T0, L0, P0, rho0
	global dm , m, r, T, L, P, rho, epsilon, F_R, F_C
	dm , m, r, T, L, P, rho, epsilon, F_R, F_C = [],[],[],[],[],[],[],[],[],[]
	
	with open('data.txt','r') as inFile:
		init_params = inFile.readline().split()
		M0 = float(init_params[0]); R0 = float(init_params[1])
		T0 = float(init_params[2]); L0 = float(init_params[3])
		P0 = float(init_params[4]); rho0 = float(init_params[5])
		for line in inFile:
			data = line.strip().split()
			dm.append(data[0])
			m.append(data[1])
			r.append(data[2])
			T.append(data[3])
			L.append(data[4])
			P.append(data[5])
			rho.append(data[6])
			epsilon.append(data[7])
			F_R.append(data[8])
			F_C.append(data[9])

	dm = np.asarray(dm, dtype=np.float64)
	m = np.asarray(m, dtype=np.float64)
	r = np.asarray(r, dtype=np.float64)
	T = np.asarray(T, dtype=np.float64)
	L = np.asarray(L, dtype=np.float64)
	P = np.asarray(P, dtype=np.float64)
	rho = np.asarray(rho, dtype=np.float64)
	epsilon = np.asarray(epsilon, dtype=np.float64)
	F_R = np.asarray(F_R, dtype=np.float64)
	F_C = np.asarray(F_C, dtype=np.float64)

	return None
read()

def plots():
	"""
	Function that plots the physical parameters.
	"""

	# Setting axis labels etc. in plots to the LaTeX font.
	plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
	params = {'text.usetex' : True,
			'font.size' : 22,
			'font.family' : 'lmodern',
			'text.latex.unicode': True,
			}
	plt.rcParams.update(params)
	
	# Plotting F_R(r) and F_C(r)
	fig_F = plt.figure()
	ax_F = fig_F.add_subplot(111)

	ax_F.set_title('Energy transport, $R_0 =$ %3.0g$R_{\odot}$'%(R0/R_sun))
	ax_F.set_xlabel('$r/R_0$')
	ax_F.set_ylabel('Relative flux')
	ax_F.plot(r,F_R, label='$F_{/mathrm{R}}$')
	ax_F.hold('on')
	ax_F.plot(r,F_C, label='$F_{\mathrm{C}}$')
	ax_F.hold('off')
	ax_F.grid('on')
	fig_F.tight_layout()

	# Plotting m(r) 
	fig_m = plt.figure()
	ax_m = fig_m.add_subplot(111)

	ax_m.set_title('Mass, $M_0 = M_{\odot}$, $R_0 =$ %3.0g$R_{\odot}$'%(R0/R_sun))
	ax_m.set_xlabel('$r/R_0$')
	ax_m.set_ylabel('$m/M_0$')
	ax_m.plot(r,m)
	ax_m.grid('on')
	fig_m.tight_layout()
	
	# Plotting P(r)
	fig_P = plt.figure()
	ax_P = fig_P.add_subplot(111)

	ax_P.set_title('Pressure, $P_0 =$ %3.2e Ba' % P0)
	ax_P.set_xlabel('$r/R_0$')
	ax_P.set_ylabel('$P/P_0$')
	ax_P.plot(r,P)
	ax_P.grid('on')
	fig_P.tight_layout()

	# Plotting L(r)
	fig_L = plt.figure()
	ax_L = fig_L.add_subplot(111)

	ax_L.set_title('Luminosity, $L_0 = L_{\odot}$')
	ax_L.set_xlabel('$r/R_0$')
	ax_L.set_ylabel('$L/L_0$')
	ax_L.plot(r,L)
	ax_L.grid('on')
	fig_L.tight_layout()

	# Plotting T(r)
	fig_T = plt.figure()
	ax_T = fig_T.add_subplot(111)

	ax_T.set_title('Temperature, $T_0 =$ %4.0f K' % T0)
	ax_T.set_xlabel('$r/R_0$')
	ax_T.set_ylabel('$T/T_0$')
	ax_T.plot(r,T)
	ax_T.grid('on')
	fig_T.tight_layout()

	# Plotting rho(r)
	fig_p = plt.figure()
	ax_p = fig_p.add_subplot(111)

	ax_p.set_title('Density, $\\rho_0 =$ %3.0f' % rho[0])
	ax_p.set_xlabel('$r/R_0$')
	ax_p.set_ylabel('$\\rho/\\rho_0$')
	ax_p.plot(r,rho)
	ax_p.grid('on')
	fig_p.tight_layout()

	# Plotting epsilon(T, rho)
	fig_e = plt.figure()
	ax_e = fig_e.add_subplot(111)

	ax_e.set_title('Energy generation per mass unit')
	ax_e.set_xlabel('$r/R_0$')
	ax_e.set_ylabel('$\\varepsilon$ [erg $g^{-1}$]')
	ax_e.plot(r,epsilon)
	ax_e.grid('on')
	fig_e.tight_layout()

	plt.show()

	pass

plots()
	
