def plot(clf, X, y):
    import matplotlib.pyplot as plt
    import pylab as pl

    x_min, x_max = min(X[:,0]), max(X[:,0])
    y_min, y_max = min(X[:,1]), max(X[:,1])

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    h = .01  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    plt.pcolormesh(xx, yy, Z, cmap=pl.cm.binary)#,alpha=0.5)
    '''
    # Plot also the test points
    grade_sig = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==0]
    bumpy_sig = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==0]
    grade_bkg = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==1]
    bumpy_bkg = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==1]
    '''

    l0 = np.array([X[i] for i in range(len(X)) if y[i] == 0.0])
    l1 = np.array([X[i] for i in range(len(X)) if y[i] == 1.0])

    plt.scatter(l0[:,0], l0[:,1], color = "b", s=1)
    plt.scatter(l1[:,0], l1[:,1], color = "r", s=1)
    plt.show()
