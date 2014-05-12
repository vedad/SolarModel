import numpy as np
import matplotlib.pyplot as plt

def read():
	
	global R_sun; R_sun = 6.96e10
	global M0, R0, T0, L0, P0, rho0
	global dm , m, r, T, L, P, rho, epsilon, F_R, F_C, PP1, PP2, PP3
	dm,m,r,T,L,P,rho,epsilon,F_R,F_C,PP1,PP2,PP3 = [],[],[],[],[],[],[],[],[],[],[],[],[]
	
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
			PP1.append(data[10])
			PP2.append(data[11])
			PP3.append(data[12])

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
	PP1 = np.asarray(PP1, dtype=np.float64)
	PP2 = np.asarray(PP2, dtype=np.float64)
	PP3 = np.asarray(PP3, dtype=np.float64)

	return None
read()

def plots():
	"""
	Function that plots the physical parameters.
	"""

	# Setting axis labels etc. in plots to the LaTeX font.
	plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
	params = {'text.usetex' : True,
			'font.size' : 20,
			'font.family' : 'lmodern',
			'text.latex.unicode': True,
			}
	plt.rcParams.update(params)
	
	# Plotting dm(r)
	fig_dm = plt.figure()
	ax_dm = fig_dm.add_subplot(111)

	ax_dm.set_title('Evolution of mass step')
	ax_dm.set_xlabel('$r/R_0$')
	ax_dm.set_ylabel('abs(d$m$)')
	ax_dm.plot(r,abs(dm))
	ax_dm.grid('on')
	fig_dm.tight_layout()

	# Plotting F_R(r) and F_C(r)
	fig_F = plt.figure()
	ax_F = fig_F.add_subplot(111)

	ax_F.set_title('$R_0 =$ %4.0f$R_{\odot}$'%(R0/R_sun))
	ax_F.set_xlabel('$r/R_0$')
	ax_F.set_ylabel('Relative flux')
	ax_F.plot(r,F_R, label='$F_{\mathrm{R}}$')
	ax_F.hold('on')
	ax_F.plot(r,F_C, label='$F_{\mathrm{C}}$')
	ax_F.hold('off')
	ax_F.grid('on')
	ax_F.legend(loc='best')
	fig_F.tight_layout()

	# Plotting m(r) 
	fig_m = plt.figure()
	ax_m = fig_m.add_subplot(111)

	ax_m.set_title('$M_0 = 1.5M_{\odot}$, $R_0 =$ %3.0g$R_{\odot}$'%(R0/R_sun))
	ax_m.set_xlabel('$r/R_0$')
	ax_m.set_ylabel('$m/M_0$')
	ax_m.plot(r,m)
	ax_m.grid('on')
	fig_m.tight_layout()
	
	# Plotting P(r)
	fig_P = plt.figure()
	ax_P = fig_P.add_subplot(111)

	ax_P.set_title('$P_0 =$ %2.1e Ba, $M_0 = 1.5M_{\odot}$, $R_0 =$ %3.0g$R_{\odot}$'%(P0, R0/R_sun))
	ax_P.set_xlabel('$r/R_0$')
	ax_P.set_ylabel('$\log_{10}(P/P_0)$')
	ax_P.plot(r,np.log10(P))
	ax_P.grid('on')
	fig_P.tight_layout()

	# Plotting L(r)
	fig_L = plt.figure()
	ax_L = fig_L.add_subplot(111)

	ax_L.set_title('$L_0 = L_{\odot}$, $M_0 = 1.5M_{\odot}$, $R_0 =$ %3.0g$R_{\odot}$'%(R0/R_sun))
	ax_L.set_xlabel('$r/R_0$')
	ax_L.set_ylabel('$\log_{10}(L/L_0)$')
	ax_L.plot(r,np.log10(L))
	ax_L.grid('on')
	fig_L.tight_layout()

	# Plotting T(r)
	fig_T = plt.figure()
	ax_T = fig_T.add_subplot(111)

	ax_T.set_title('$T_0 =$ %4.0f K, $M_0 = 1.5M_{\odot}$ ,$R_0 =$ %3.0g$R_{\odot}$'%(T0,R0/R_sun))
	ax_T.set_xlabel('$r/R_0$')
	ax_T.set_ylabel('$\log_{10}(T/T_0)$')
	ax_T.plot(r,np.log10(T))
	ax_T.grid('on')
	fig_T.tight_layout()

	# Plotting rho(r)
	fig_p = plt.figure()
	ax_p = fig_p.add_subplot(111)

	ax_p.set_title('Density, $\\rho_0 =$ %1.1e g $\mathrm{cm}^{-3}$' % rho0)
	ax_p.set_xlabel('$r/R_0$')
	ax_p.set_ylabel('$\\rho/\\rho_0$')
	ax_p.plot(r,rho)
	ax_p.grid('on')
	fig_p.tight_layout()

	# Plotting epsilon(T, rho)
	fig_e = plt.figure()
	ax_e = fig_e.add_subplot(111)

	ax_e.set_title('$R_0 =$ %3.0f$R_{\odot}$'%(R0/R_sun))
	ax_e.set_xlabel('$r/R_0$')
	ax_e.set_ylabel('$\\varepsilon$ [erg $g^{-1}$]')
	ax_e.plot(r,epsilon, label='$\\varepsilon$')
	ax_e.plot(r,PP1, label='PP1')
	ax_e.hold('on')
	ax_e.plot(r,PP2, label='PP2')
	ax_e.plot(r,PP3, label='PP3')
	ax_e.hold('off')
	ax_e.grid('on')
	ax_e.legend(loc='best')
	fig_e.tight_layout()

	plt.show()

	pass

plots()
	
