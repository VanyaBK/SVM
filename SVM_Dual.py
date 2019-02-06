def solve_SVM_dual_CVXOPT(x_train, y_train, x_test, C=1):
    """
    Solves the SVM training optimisation problem (the dual) using cvxopt.
    Arguments:
    x_train: A numpy array with shape (n,d), denoting n training samples in \R^d. 
    y_train: numpy array with shape (n,) Each element takes +1 or -1
    x_train: A numpy array with shape (m,d), denoting m test samples in \R^d. 
    C : The tradeoff parameter in the training objective
    
    Limits:
    n<200, d<100000, m<1000
    
    
    Returns:
    y_test : A numpy array with shape (m,). This is the result of running the learned model on the
    test instances x_test. Each element is  +1 or -1.
    
    alphas_1 : A numpy array with shape (n,) giving the Lagrange multipliers obtained by solving the dual.
    
    """
    n = x_train.shape[0]
    #Solving the dual
    K = y_train[:, None] * x_train
    K = np.dot(K, K.T)
    P = matrix(K)
    q = -1*matrix(np.ones((n, 1)))
    G = -1*matrix(np.eye(n))
    h = matrix(np.zeros(n))
    A = matrix(y_train.reshape(1, -1))
    b = matrix(np.zeros(1))
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    #getting weights
    w = np.sum(alphas * y_train[:, None] * x_train, axis = 0)
    # getting bias
    cond = (alphas > 1e-4).reshape(-1)
    b = y_train[cond] - np.dot(x_train[cond], w)
    bias = b[0]
    for i in range(x_test.shape[0]):
        y_test[i] = np.dot(w.T,x_test[i])+bias
        if(y_test[i]>=0):
            y_test[i] = 1
        else:
            y_test[i] = -1
    #Lagrange Multipliers
    alphas = alphas.reshape(n,)
    alphas_1 = np.zeros(n,)
    for i in range(n):
        if(alphas[i]>=0 and alphas[i]<=C):
            alphas_1[i] = alphas[i]
    return (y_test,alphas_1)
    
	
import random as rnd
def solve_SVM_dual_SMO(x_train, y_train, x_test, C=1):
    """
    Solves the SVM training optimisation problem (the dual) using Sequential Minimal Optimisation.
    Arguments:
    x_train: A numpy array with shape (n,d), denoting n training samples in \R^d. 
    y_train: numpy array with shape (n,) Each element takes +1 or -1
    x_train: A numpy array with shape (m,d), denoting m test samples in \R^d. 
    C : The tradeoff parameter in the training objective
    Limits:
    n<200, d<100000, m<1000
    
    
    Returns:
    y_pred_test : A numpy array with shape (m,). This is the result of running the learned model on the
    test instances x_test. Each element is  +1 or -1.
    
    alpha : A numpy array with shape (n,) giving the Lagrange multipliers obtained by solving the dual.

    """
    n, d = x_train.shape[0], x_train.shape[1]
    alpha = np.zeros((n))
    count = 0
    while True:
        count += 1
        alpha_prev = np.copy(alpha)
        for j in range(0, n):
            # Getting random int i!=j
            i = j
            cnt=0
            while i == j and cnt<1000:
                i = rnd.randint(0,n-1)
                cnt=cnt+1
            x_i, x_j, y_i, y_j = x_train[i,:], x_train[j,:], y_train[i], y_train[j]
            k_ij = (np.dot(x_i, x_i.T)) + (np.dot(x_j, x_j.T) ) - (2 * np.dot(x_i, x_j.T))
            if k_ij <= 0:
                continue
            alpha_prime_j, alpha_prime_i = alpha[j], alpha[i]
            if(y_i != y_j):
                (L,H) = (max(0, alpha_prime_j - alpha_prime_i), min(C, C - alpha_prime_i + alpha_prime_j))
            else:
                (L,H) = (max(0, alpha_prime_i + alpha_prime_j - C), min(C, alpha_prime_i + alpha_prime_j))
            if(L==H):
                continue
            # Computing model parameters
            w = np.dot(x_train.T, np.multiply(alpha,y_train))
            b = np.mean(y_train - np.dot(w.T, x_train.T))
            E_i = np.sign(np.dot(w.T, x_i.T) + b).astype(int) - y_i
            E_j = np.sign(np.dot(w.T, x_j.T) + b).astype(int) - y_j
            # Setting new alpha values(Lagrange multipliers)
            alpha[j] = alpha_prime_j + float(y_j * (E_i - E_j))/k_ij
            alpha[j] = max(alpha[j], L)
            alpha[j] = min(alpha[j], H)
            alpha[i] = alpha_prime_i + y_i*y_j * (alpha_prime_j - alpha[j])
        # Checking for convergence
        diff = np.linalg.norm(alpha - alpha_prev)
        if diff < 0.000000001:
            break
    # Computing weights and bias
    b = np.mean(y_train-np.dot(w.T,x_train.T))
    w = np.dot(x_train.T, np.multiply(alpha,y_train))
    y_pred_test = (np.sign(np.dot(w.T, x_test.T) + b).astype(int))
    return (y_pred_test,alpha)
    
