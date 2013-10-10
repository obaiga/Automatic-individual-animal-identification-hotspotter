import matplotlib
if matplotlib.get_backend() != 'Qt4Agg':
    matplotlib.use('Qt4Agg', warn=True, force=True)
    matplotlib.rcParams['toolbar'] = 'None'
import hotspotter.draw_func2 as df2
import hotspotter.helpers as helpers
import matplotlib.pyplot as plt

import numpy as np
import scipy

import matplotlib
import matplotlib.pyplot as plt

def numpy_test():
    # Constants
    inv   = scipy.linalg.inv
    sqrtm = scipy.linalg.sqrtm
    tau = np.pi*2
    sin = np.sin
    cos = np.cos
    # Variables
    N = 2**5
    theta = np.linspace(0, tau, N)
    a, b, c, d = (1, 0, .5, .8)

    xc = np.array((sin(theta), cos(theta))) # points on unit circle 2xN

    A = np.array([(a, b),   # maps points on ellipse
                  (c, d)])  # to points on unit circle

    # Test data
    Ainv = inv(A)

    # Test data
    xe = Ainv.dot(xc)

    # Test Ellipse
    E = A.T.dot(A) # equation of ellipse 
    test = lambda ell: ell.T.dot(E).dot(ell).diagonal()

    print all(np.abs(1 - test(xe)) < 1e-9)

    # Start Plot
    df2.reset()
    def plotell(ell):
        df2.plot(ell[0], ell[1])
    # Plot Circle
    fig = df2.figure(1, plotnum=121, title='xc')
    plotell(xc)
    # Plot Ellipse
    fig = df2.figure(1, plotnum=122, title='xe = inv(A).dot(xc)')
    plotell(xe)
    # End Plot
    df2.set_geometry(fig.number, 1000, 75, 500, 500)
    df2.update()

# E = ellipse
#
# points x on ellipse satisfy x.T * E * x = 1
#
# A.T * A = E

# A = transforms points on an ellipse to a unit circle
def sqrtm_eq():
    M = np.array([(33, 24), (48, 57)])
    R1 = np.array([(1, 4), (8, 5)])
    R2 = np.array([(5, 2), (4, 7)])
    print M
    print R1.dot(R1)
    print R2.dot(R2)
    '''

    matrix is positive semidefinite 
    if x.conjT.dot(M).dot(x) >= 0

    or if rank(A) = rank(A.conjT.dot(A))
    '''

def find_ellipse_major_minor():
    import sympy
    import sympy.galgebra.GA as GA
    import sympy.galgebra.latex_ex as tex
    import sympy.printing as symprint
    import sympy.abc
    import sympy.mpmath
    Eq = sympy.Eq
    solve = sympy.solve
    
    sin = sympy.functions.elementary.trigonometric.sin
    cos = sympy.functions.elementary.trigonometric.cos

    a, b, c, d = sympy.symbols('a b c d')
    theta = sympy.abc.theta
    # R * E
    a2 = cos(theta)*a - sin(theta)*c 
    b2 = cos(theta)*b - sin(theta)*d
    c2 = sin(theta)*a + cos(theta)*c
    d2 = sin(theta)*b + cos(theta)*d

    # Set b2 and c2 to 0
    b2_eq_0 = Eq(b2,0)
    #
    c2_eq_0 = Eq(c2,0)

    theta_b = solve(b2_eq_0, theta)
    theta_c = solve(c2_eq_0, theta)

def sympy_manual_sqrtm_inv():
    '''
    Manual calculation of inv(sqrtm) for a lower triangular matrix
    '''
    # https://sympy.googlecode.com/files/sympy-0.7.2.win32.exe
    import sympy
    import sympy.galgebra.GA as GA
    import sympy.galgebra.latex_ex as tex
    import sympy.printing as symprint
    import sympy.abc
    import sympy.mpmath

    a, b, c, d = sympy.symbols('a b c d')
    theta = sympy.abc.theta
    sin = sympy.functions.elementary.trigonometric.sin
    cos = sympy.functions.elementary.trigonometric.cos
    sqrtm = sympy.mpmath.sqrtm

    xc = sympy.Matrix([sin(theta), cos(theta)])
    A = sympy.Matrix([(a, b), (c, d)])
    A = sympy.Matrix([(a, 0), (c, d)])
    Ainv = A.inv()

    #Asqrtm = sqrtm(A)

    def asMatrix(list_): 
        N = int(len(list_) / 2)
        Z = sympy.Matrix([list_[0:N], list_[N:]])
        return Z

    def simplify_mat(X):
        list_ = [_.simplify() for _ in X]
        Z = asMatrix(list_)
        return Z

    def symdot(X, Y):
        Z = asMatrix(X.dot(Y))
        return Z

    Eq = sympy.Eq
    solve = sympy.solve

    a, b, c, d = sympy.symbols('a b c d')
    w, x, y, z = sympy.symbols('w x y z')

    R = sympy.Matrix([(w, x), (y, z)])
    M = symdot(R,R)

    # Solve in terms of A
    eq1 = Eq(a, M[0])
    eq2 = Eq(c, M[2])
    eq3 = Eq(d, M[3])
    w1 = solve(eq1, w)[1].subs(y,0)
    #y1 = sympy.Eq(0, M[1]) # y is 0
    x1 = solve(eq2, x)[0]
    z1 = solve(eq3, z)[1].subs(y,0)
    x2 = x1.subs(w, w1).subs(z, z1)

    R_itoA = simplify_mat(sympy.Matrix([(w1, x2), (0, z1)]))
    Rinv_itoA = simplify_mat(R_itoA.inv())

    print('Given the lower traingular matrix: A=[(a, 0), (c, d)]')

    print('Its inverse is: inv(A)')
    print(Ainv) # sub to lower triangular
    print('--')

    print('Its square root is: R = sqrtm(A)')
    print(R_itoA)
    print('--')

    print('Method 1: Its inverse square root is: M1 = inv(sqrtm(A))')
    print(Rinv_itoA)
    print('--')

    # Solve in terms of A (but from inv)
    a_, b_, c_, d_ = Ainv
    eq1_ = Eq(a_, M[0])
    eq2_ = Eq(c_, M[2])
    eq3_ = Eq(d_, M[3])
    w1_ = solve(eq1_, w)[1].subs(y,0)
    x1_ = solve(eq2_, x)[0]
    z1_ = solve(eq3_, z)[1].subs(y,0)
    #sympy.Eq(0, M[1]) # y is 0
    x2_ = x1_.subs(w, w1_).subs(z, z1_)

    Rinv_itoA_2 = simplify_mat(sympy.Matrix([(w1_, x2_), (0, z1_)]))

    print('Method 1: Its square root inverse is: M2 = sqrtm(inv(A))')
    print(Rinv_itoA_2)
    print('----')

    print('Perform checks to make sure the calculation was correct')
    print('Checking that A == inv(M1 M1)')
    print simplify_mat(symdot(Rinv_itoA,Rinv_itoA).inv())
    print('----')
    print('Checking that A == inv(M2 M2)')
    print simplify_mat(symdot(Rinv_itoA_2,Rinv_itoA_2).inv())

    print('....')
    print('....')
    # hmmm, why not equal? ah, they are equiv
    sqrt = sympy.sqrt
    ans1 = c/(-sqrt(a)*d - a*sqrt(d))
    ans2 = -c/(a*d*(sqrt(1/a) + sqrt(1/d)))
    print('There are two equivilent ways of specifying the b component of sqrt(inv(A))')
    print ans1
    print ans2
     #------------------------------
    A2 = simplify_mat(R_itoA.dot(R_itoA))

    E_ = A.T.dot(A)
    E = sympy.Matrix([E_[0:2], E_[2:4]])
    print('Original ellipse matrix: E = A.T.dot(A)')
    print(E.subs(b,0))

    E_evects = E.eigenvects()
    E_evals = E.eigenvals()

    e1, e2 = E_evects
    print('\n---Eigenvect1---')
    print('\ne1[0]=')
    print e1[0]
    print('\ne1[1]=')
    print e1[1]
    print('\ne1[2]=')
    print e1[2]
    print('\n---Eigenvect2---')
    print('\ne2[0]=')
    print e2[0]
    print('\ne2[1]=')
    print e2[1]
    print('\ne2[2]=')
    print e2[2]
    print('\n---(inv(sqrtm(A))) dot circle points---')
    print('Given a transformation A and an angle theta, the point on that ellipse is: ')
    xe = Rinv_itoA.dot(xc)
    print xe


#A2 = sqrtm(inv(A)).real
#A3 = inv(sqrtm(A)).real
#fig = df2.figure(1, plotnum=223, title='')
#plotell(e2)
#fig = df2.figure(1, plotnum=224, title='')
#plotell(e3)

#df2.reset()
#real_data()
sympy_manual_sqrtm_inv()

r'''
str1 = '\n'.join(helpers.execstr_func(sympy_data).split('\n')[0:-2])
str2 = '\n'.join(helpers.execstr_func(execute_test).split('\n')[3:])

toexec = str1 + '\n' + str2
exec(toexec)
'''