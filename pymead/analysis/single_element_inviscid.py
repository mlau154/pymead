import numpy as np
from numba import jit
from numpy import zeros, pi, arctan2, sin, cos, sqrt, log, dot
from numpy.linalg import inv


@jit(nopython=True, cache=True)
def single_element_inviscid(coord: np.ndarray, alpha: float or np.float64):
    r"""
    A linear strength vortex panel method for the inviscid solution of a single airfoil, sped up using the
    just-in-time compiler in *numba*. Directly adapted from "Program 7" of Appendix D in [1].

    [1] J. Katz and A. Plotkin, Low-Speed Aerodynamics, Second Edition, 2nd ed. New York, NY,
    USA: Cambridge University Press, 2004. Accessed: Mar. 07, 2023. [Online].
    Available: `<https://asmedigitalcollection.asme.org/fluidsengineering/article/126/2/293/458666/LowSpeed-Aerodynamics-Second-Edition>`_

    Parameters
    ----------
    coord: np.ndarray
        An :math:`N \times 2` array of airfoil coordinates, where :math:`N` is the number of coordinates, and the columns
        represent :math:`x` and :math:`y`

    alpha: float or np.float64
        Angle of attack of the airfoil

    Returns
    -------
    np.ndarray, np.ndarray, float
        The first returned array is of size :math:`(N-1) \times 2` and represents the :math:`x`- and :math:`y`-locations
        of the collocation points, where :math:`N` is the number of airfoil coordinates. The second returned array is a
        one-dimensional array with length :math:`(N-1)` representing the surface pressure coefficient at each collocation
        point. The final returned value is the lift coefficient.
    """

    N = len(coord)          # Number of panel end points
    M = N - 1                 # Number of control points

    EP = zeros((N, 2))      # Clockwise-defined panel end points
    EPT = zeros((N, 2))     # End points from file
    PT1 = zeros((M, 2))     # Start point of panel
    PT2 = zeros((M, 2))     # End point of panel
    CO = zeros((M, 2))      # Collocation point
    A = zeros((N, N))       # Aerodynamic influence coefficient matrix
    B = zeros((N, N))       # Tangential induced velocities (with gammas)
    TH = zeros((M,))      # Panel angle
    DL = zeros((M,))      # Panel length
    RHS = zeros((N, 1))     # Freestream component normal to panel
    V = zeros((M,))       # Panel tangential velocity

    ALPHA = alpha  # AOA in deg
    AL = ALPHA * pi / 180

    EPT[:, 0] = coord[:, 0]    # Read in x/c position of panel end points
    EPT[:, 1] = coord[:, 1]    # Read in y/c position of panel end points

    # Order panel end points defined in clockwise direction
    for i in range(N):
        EP[i, 0] = EPT[N - i - 1, 0]
        EP[i, 1] = EPT[N - i - 1, 1]

    # Define end points of each panel (PT1 is beginning point, PT2 is end point)
    for i in range(M):
        PT1[i, 0] = EP[i, 0]
        PT2[i, 0] = EP[i+1, 0]
        PT1[i, 1] = EP[i, 1]
        PT2[i, 1] = EP[i+1, 1]

    # Determine local slope of each panel
    for i in range(M):
        DZ = PT2[i, 1] - PT1[i, 1]
        DX = PT2[i, 0] - PT1[i, 0]
        TH[i] = arctan2(DZ, DX)

    # Identify collocation points for each panel (half-panel location)
    for i in range(M):
        CO[i, 0] = (PT2[i, 0] - PT1[i, 0]) / 2 + PT1[i, 0]
        CO[i, 1] = (PT2[i, 1] - PT1[i, 1]) / 2 + PT1[i, 1]

    # Determine influence coefficients
    for i in range(M):
        for j in range(M):

            # Determine location of collocation point i in terms of panel j
            # coordinates
            XT = CO[i, 0] - PT1[j, 0]
            ZT = CO[i, 1] - PT1[j, 1]
            X2T = PT2[j, 0] - PT1[j, 0]
            Z2T = PT2[j, 1] - PT1[j, 1]

            X = XT*cos(TH[j]) + ZT*sin(TH[j])
            Z = -XT*sin(TH[j]) + ZT*cos(TH[j])
            X2 = X2T*cos(TH[j]) + Z2T*sin(TH[j])
            Z2 = 0

            # Store length of each panel (only required for first loop in i)
            if i == 0:
                DL[j] = X2

            # Determine radial distance and angle between corner points of jth
            # panel and ith control point
            R1 = sqrt(X**2 + Z**2)
            R2 = sqrt((X - X2)**2 + Z**2)
            TH1 = arctan2(Z, X)
            TH2 = arctan2(Z, X - X2)

            # Determine influence coefficient of jth panel on ith control point
            # (include consideration for self-induced velocities)
            if i == j:
                U1L = -0.5*(X - X2) / X2
                U2L = 0.5*X / X2
                W1L = -0.15916
                W2L = 0.15916
            else:
                U1L = -(Z*log(R2/R1) + X*(TH2 - TH1) - X2*(TH2 - TH1)) / (6.28319*X2)
                U2L = (Z*log(R2/R1) + X*(TH2 - TH1)) / (6.28319*X2)
                W1L = -((X2 - Z*(TH2 - TH1)) - X*log(R1/R2) + X2*log(R1/R2)) / (6.28319*X2)
                W2L = ((X2 - Z*(TH2 - TH1)) - X*log(R1/R2)) / (6.28319*X2)

            # Rotate coordinates back from jth panel reference frame to airfoil
            # chord frame
            U1 = U1L * np.cos(-TH[j]) + W1L * np.sin(-TH[j])
            U2 = U2L*cos(-TH[j]) + W2L*sin(-TH[j])
            W1 = -U1L*sin(-TH[j]) + W1L*cos(-TH[j])
            W2 = -U2L*sin(-TH[j]) + W2L*cos(-TH[j])

            # Define AIC: A(i,j) is the component of velocity normal to control
            # point i due to panel j
            # B(i,j) is the tangential velocity along control point i due to
            # panel j, used after solving for gammas
            if j == 0:
                A[i, 0] = -U1*sin(TH[i]) + W1*cos(TH[i])
                HOLDA = -U2*sin(TH[i]) + W2*cos(TH[i])
                B[i, 0] = U1*cos(TH[i]) + W1*sin(TH[i])
                HOLDB = U2*cos(TH[i]) + W2*sin(TH[i])
            elif j == M - 1:
                A[i, M - 1] = -U1*sin(TH[i]) + W1*cos(TH[i]) + HOLDA
                A[i, N - 1] = -U2*sin(TH[i]) + W2*cos(TH[i])
                B[i, M - 1] = U1*cos(TH[i]) + W1*sin(TH[i]) + HOLDB
                B[i, N - 1] = U2*cos(TH[i]) + W2*sin(TH[i])
            else:
                A[i, j] = -U1*sin(TH[i]) + W1*cos(TH[i]) + HOLDA
                HOLDA = -U2*sin(TH[i]) + W2*cos(TH[i])
                B[i, j] = U1*cos(TH[i]) + W1*sin(TH[i]) + HOLDB
                HOLDB = U2*cos(TH[i]) + W2*sin(TH[i])

        # Set up freestream component of boundary condition
        RHS[i, 0] = cos(AL)*sin(TH[i]) - sin(AL)*cos(TH[i])

    # Enforce Kutta condition
    RHS[N - 1, 0] = 0

    A[N - 1, 0] = 1
    A[N - 1, N - 1] = 1

    # Invert A matrix to solve for gammas
    G = dot(inv(A), RHS)

    # With known gammas, solve for CL, CPs
    CL = 0.0

    for i in range(M):
        VEL = 0.0
        for j in range(N):
            VEL = VEL + (B[i, j] * G[j])[0]
        V[i] = VEL + cos(AL)*cos(TH[i]) + sin(AL)*sin(TH[i])
        CL = CL + ((G[i] + G[i + 1]) * DL[i])[0]

    CP = 1 - V**2

    return CO, CP, CL
