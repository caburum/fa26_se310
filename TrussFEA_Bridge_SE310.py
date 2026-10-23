"""
===============================================================================
 SE 310 - Project I : TRUSS BRIDGE DESIGN CHALLENGE
 Direct Stiffness (FEA) solver for 2-D pin-jointed trusses
-------------------------------------------------------------------------------

 UNITS USED THROUGHOUT:  length = mm ,  force = N ,  stress & modulus = MPa=N/mm^2
 (Using mm-N-MPa keeps every number in a readable range for a model bridge.)

 WHAT YOU HAVE TO EDIT:  only the block marked  "1. PROBLEM DEFINITION".
===============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
#  0. MATERIAL AND SECTION PROPERTIES  (Midwest 1/8 in. square balsa stick)
# =============================================================================
b_stick   = 3.18          # mm   - stick width  (1/8 inch)
h_stick   = 3.18          # mm   - stick depth  (1/8 inch)
E_balsa   = 3000.0        # MPa  - Young's modulus along the grain (3 GPa)
sig_t_ult = 7.5           # MPa  - ultimate TENSILE  strength along the grain
sig_c_ult = 4.5           # MPa  - ultimate CRUSHING strength along the grain
rho_balsa = 160e-9        # kg/mm^3 - density (160 kg/m^3)
FS_req    = 2.0           # [-]  - required factor of safety (project constraint)
K_eff     = 1.0           # [-]  - effective length factor, pin-pin ends

A_stick = b_stick * h_stick                     # mm^2  cross-sectional area
I_stick = b_stick * h_stick**3 / 12.0           # mm^4  weak-axis second moment

# -----------------------------------------------------------------------------
#  LOADING  -  the two course textbooks resting ON TOP of the bridge
#  >>> WEIGH YOUR OWN BOOKS AND REPLACE THESE NUMBERS <<<
# -----------------------------------------------------------------------------
m_book_SA  = 1.50         # kg - "Structural Analysis"
m_book_MM  = 1.00         # kg - "Machines and Mechanisms"
g          = 9.81         # m/s^2
W_books    = (m_book_SA + m_book_MM) * g        # N  - total service load
P_design   = FS_req * W_books                   # N  - factored DESIGN load
N_TRUSS    = 3                                  # two identical vertical trusses
P_truss    = P_design / N_TRUSS                 # N  - design load on ONE truss

# =============================================================================
#  1. PROBLEM DEFINITION   <-- THIS IS THE ONLY BLOCK YOU NEED TO CHANGE
# =============================================================================
#  SAMPLE DESIGN : symmetric 4-panel PRATT DECK-truss, one vertical side truss.
#  The two textbooks rest ON TOP of the bridge, so the TOP chord is the loaded
#  deck. Span 200 mm, height 70 mm (the envelope allows up to 100 mm above the desk).
#
#   T0----T1----T2----T3----T4      y = 70 mm   books rest on this chord
#   | \    | \   |   / |   / |
#   |  \   |  \  |  /  |  /  |
#   B0----B1----B2----B3----B4      y =  0 mm   desk level
#   ^                       ^
#  pin                    roller    <-- the two desk edges, 200 mm apart
#
D_truss = 70.0            # mm - truss height above the desk top ( <= 100 mm )
SPAN    = 250.0           # mm - clear span = overall length of the truss
x_sup_L = 0.0             # mm - left support (edge of the left desk)
x_sup_R = SPAN           # mm - right support (edge of the right desk)

# ---- Nodal coordinates [x, y] (node numbering starts at 0, as in the original)
nodeCords = np.array([
	[SPAN/4,	D_truss],   # 0  T0 - upper left
	[SPAN*3/4,	D_truss],   # 1  T1 - upper right
	[0.0,		0.0],   # 2  B0 - LEFT SUPPORT  (pin)
	[SPAN/2,	0.0],   # 3  B1
	[SPAN,		0.0],   # 4  B2 - RIGHT SUPPORT (roller)
])
print("nodeCords = ")
print(nodeCords)

# ---- Element connectivity [near node, far node]
elemNodes = np.array([
	# --- top chord = the deck the books sit on -----------------------------
	[0, 1],
	# --- bottom chord (at desk level) --------------------------------------
	[2, 3], [3, 4],
	# --- verticals (the end verticals sit directly over the supports) ------
	# --- Pratt diagonals (slope DOWN towards mid-span -> they take tension) -
	[0, 2], [0, 3], [1, 3], [1, 4]
])

# ---- Material and section of every element
modE = np.full((elemNodes.shape[0], 1), E_balsa)    # MPa
Area = np.full((elemNodes.shape[0], 1), A_stick)    # mm^2
Imom = np.full((elemNodes.shape[0], 1), I_stick)    # mm^4

# ---- Displacement constraints [node, dof, value]   dof: 1 = x , 2 = y
DispCon = np.array([
	[2, 1, 0.0],      # node 2 : pinned  -> ux = 0
	[2, 2, 0.0],      # node 2 : pinned  -> uy = 0
	[4, 2, 0.0],      # node 4 : roller  -> uy = 0
])

# ---- Applied forces [node, dof, value]   (downward = negative y)
#  The books lie on the 200 mm deck.  Half of P_design goes to each side truss
#  and is shared between the FOUR deck panels by tributary length:
#  assuming all joints take the same weight
Fval = np.array([
	[0, 2, -P_truss / 2.0],
	[1, 2, -P_truss / 2.0],
])


CSV_OUT     = 'truss_results.csv'   # where the table is written
DEFL_LIMIT  = 2.0      # mm - allowable mid-span deflection (design constraint)
MID_NODE    = 3        # mid-span deck joint whose deflection is checked # fixme:
SCALE_PLOT  = 20       # deformation magnification for the plot

# =============================================================================
#  2. PROBLEM INITIALIZATION      (unchanged from the original code)
# =============================================================================
nELEM = elemNodes.shape[0]            # number of elements
nNODE = nodeCords.shape[0]            # number of nodes
nDC   = DispCon.shape[0]              # number of constrained dofs
nFval = Fval.shape[0]                 # number of loaded dofs
NDOF  = nNODE * 2                     # total dofs = 2 per node

uDisp     = np.zeros((NDOF, 1))       # displacement vector
forces    = np.zeros((NDOF, 1))       # force vector
Stiffness = np.zeros((NDOF, NDOF))    # global stiffness matrix
Stress    = np.zeros(nELEM)           # element stress
Fint_el   = np.zeros(nELEM)           # element internal (axial) force
L_elem    = np.zeros(nELEM)           # element length
kdof      = np.zeros(nDC)             # dofs with KNOWN displacement

xx = nodeCords[:, 0]
yy = nodeCords[:, 1]

# ---- Build the prescribed-displacement array
for i in range(nDC):
	indice = DispCon[i, :]
	v = float(indice[2])
	indice = indice.astype(int)
	kdof[i] = indice[0]*2 + indice[1] - 1
	uDisp[indice[0]*2 + indice[1] - 1] = v

# ---- Build the applied-force array
for i in range(nFval):
	indice2 = Fval[i, :]
	v = float(indice2[2])
	indice2 = indice2.astype(int)
	forces[indice2[0]*2 + indice2[1] - 1] = v

kdof  = kdof.astype(int)                            # known-displacement dofs
ukdof = np.setdiff1d(np.arange(NDOF), kdof)         # unknown-displacement dofs

# ---- Static determinacy check (m + r  vs  2j)
m_r = nELEM + nDC
print("=" * 78)
print(" STATIC DETERMINACY CHECK")
print(f"   members m = {nELEM} , reactions r = {nDC} , joints j = {nNODE}")
print(f"   m + r = {m_r}   vs   2j = {2*nNODE}  ->  "
	  + ("statically DETERMINATE" if m_r == 2*nNODE
		 else ("statically INDETERMINATE" if m_r > 2*nNODE
			   else "UNSTABLE (mechanism) - fix your geometry!")))

# =============================================================================
#  FEA STEPS 1-3 : element stiffness -> transform -> assemble
# =============================================================================
for e in range(nELEM):
	indiceE = elemNodes[e, :]
	Y  = float(modE[e, 0])
	Ae = float(Area[e, 0])
	elemDOF = np.array([indiceE[0]*2, indiceE[0]*2+1,
						indiceE[1]*2, indiceE[1]*2+1]).astype(int)
	xa = xx[indiceE[1]] - xx[indiceE[0]]
	ya = yy[indiceE[1]] - yy[indiceE[0]]
	len_elem = np.sqrt(xa*xa + ya*ya)
	lx = xa / len_elem                      # lambda x
	ly = ya / len_elem                      # lambda y

	# Step 1. Elemental stiffness matrix in LOCAL coordinates
	ke = (Y*Ae/len_elem) * np.array([[1, -1], [-1, 1]])

	# Step 2. Transform from LOCAL to GLOBAL coordinates
	T  = np.array([[lx, ly, 0, 0], [0, 0, lx, ly]])
	k2 = np.matmul(T.transpose(), np.matmul(ke, T))

	# Step 3. Assemble into the GLOBAL stiffness matrix
	Stiffness[np.ix_(elemDOF, elemDOF)] += k2

# =============================================================================
#  FEA STEP 4 : partition, solve for displacements and reactions
# =============================================================================
k11 = Stiffness[np.ix_(ukdof, ukdof)]
k12 = Stiffness[np.ix_(ukdof, kdof)]
k21 = k12.transpose()
k22 = Stiffness[np.ix_(kdof,  kdof)]

f_known = forces[ukdof] - np.matmul(k12, uDisp[kdof])
uDisp[np.ix_(ukdof)] = np.linalg.solve(k11, f_known)
forces[np.ix_(kdof)] = (np.matmul(k21, uDisp[np.ix_(ukdof)]) + np.matmul(k22, uDisp[np.ix_(kdof)]))

# =============================================================================
#  FEA STEP 5 : internal forces  +  BUCKLING CHECK
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 3.6))

F_TOL   = 1e-6 * P_truss    # forces below this are treated as exactly zero
Pcr     = np.zeros(nELEM)   # Euler critical load of each member
buckles = ['N'] * nELEM     # does the member buckle?  'Y' / 'N'

for e in range(nELEM):
	indiceE = elemNodes[e, :]
	Y, Ae, Ie = float(modE[e, 0]), float(Area[e, 0]), float(Imom[e, 0])
	elemDOF = np.array(
		[indiceE[0]*2, indiceE[0]*2+1,
		indiceE[1]*2, indiceE[1]*2+1]
	).astype(int)
	xa = xx[indiceE[1]] - xx[indiceE[0]]
	ya = yy[indiceE[1]] - yy[indiceE[0]]
	len_elem = np.sqrt(xa*xa + ya*ya)
	L_elem[e] = len_elem
	lx, ly = xa/len_elem, ya/len_elem

	ke = (Y*Ae/len_elem) * np.array([[1, -1], [-1, 1]])
	T  = np.array([[lx, ly, 0, 0], [0, 0, lx, ly]])
	Fint = np.matmul(ke, np.matmul(T, uDisp[np.ix_(elemDOF)]))

	Fint_el[e] = float(Fint[1, 0])        # +ve = TENSION , -ve = COMPRESSION
	Stress[e]  = Fint_el[e] / Ae          # MPa

	# ---- Euler buckling capacity of a pin-ended balsa stick
	Pcr[e] = np.pi**2 * Y * Ie / (K_eff*len_elem)**2

	# ---- BUCKLING CHECK
	#  Fint_el is the force under the FACTORED design load, i.e. the factor of
	#  safety FS_req = 2 is already built into P_design.  So a compression
	#  member is safe as long as  |F|  stays below its Euler load P_cr.
	#  Tension and zero-force members cannot buckle.
	if Fint_el[e] < -F_TOL:                               # compression member
		buckles[e] = 'Y' if abs(Fint_el[e]) >= Pcr[e] else 'N'
	elif Fint_el[e] <= F_TOL:                             # zero-force member
		Fint_el[e] = 0.0
		Stress[e]  = 0.0

	# ---- Plot undeformed (solid) and deformed (dashed) shape
	col = ('tab:red' if Fint_el[e] < 0 else ('tab:blue' if Fint_el[e] > 0 else '0.75'))
	ax.plot(
		[xx[indiceE[0]], xx[indiceE[1]]],
		[yy[indiceE[0]], yy[indiceE[1]]],
		color=col, lw=2, zorder=2
	)
	ax.plot(
		[
			xx[indiceE[0]] + SCALE_PLOT*uDisp[indiceE[0]*2, 0],
			xx[indiceE[1]] + SCALE_PLOT*uDisp[indiceE[1]*2, 0]
		],
		[
			yy[indiceE[0]] + SCALE_PLOT*uDisp[indiceE[0]*2+1, 0],
			yy[indiceE[1]] + SCALE_PLOT*uDisp[indiceE[1]*2+1, 0]
		],
		'--', color='0.35', lw=1.0, zorder=1
	)

# =============================================================================
#  6. RESULTS  -  one table, exported to CSV
# =============================================================================
delta_mid  = abs(uDisp[MID_NODE*2 + 1, 0])
total_len  = L_elem.sum() * N_TRUSS
mass_truss = total_len * A_stick * rho_balsa * 1000.0     # grams, both trusses

res = pd.DataFrame({
	'Elem':          np.arange(nELEM),
	'i':             elemNodes[:, 0],
	'j':             elemNodes[:, 1],
	'L (mm)':        np.round(L_elem, 1),
	'F (N)':         np.round(Fint_el, 2),
	'T/C':           ['T' if f > 0 else ('C' if f < 0 else '0') for f in Fint_el],
	'P_cr (N)':      np.round(Pcr, 1),
	'Buckling?(Y/N)': buckles,
})

res.to_csv(CSV_OUT, index=False)

print(f" Design load on ONE truss = {P_truss:.2f} N, for {N_TRUSS} repeated trusses   "
	f"(FS = {FS_req:.0f} already applied, W_books = {W_books:.2f} N)")
print(" +F = tension, -F = compression.  A member buckles when |F| reaches P_cr.")
print()
print(res.to_string(index=False))
print()
print(f" Members that buckle: {buckles.count('Y')} of {nELEM}"
	f"{'' if buckles.count('Y') == 0 else '  -> ' + str([k for k in range(nELEM) if buckles[k] == 'Y'])}")
print(f" Maximum deflection D_max = {delta_mid:.3f} mm      "
	f"Total length of material used = {total_len:.1f} mm      "
	f"Structure mass M = {mass_truss:.1f} g")
print(f" Table exported to: {CSV_OUT}")

# ---- Figure -----------------------------------------------------------------
ax.set_title(
	f"Sample design - Pratt deck-truss, books on top  "
	f"(deformation x{SCALE_PLOT}; red = compression, blue = tension)",
	fontsize=10
)
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-25, SPAN + 25); ax.set_ylim(-30, 130)
ax.grid(alpha=.25)
# support symbols
ax.plot(x_sup_L, 0, marker='^', ms=11, color='k', zorder=5)
ax.plot(x_sup_R, 0, marker='o', ms=9,  color='k', zorder=5)
fig.tight_layout()
plt.show()
