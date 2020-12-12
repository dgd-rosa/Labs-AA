# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:36:04 2019

@author: ist
"""

import numpy as np
import matplotlib.pyplot as plt
import math


# =============================================================================
# Visualization Functions
# =============================================================================


def plot_contours(clf, points,labels):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    clf: a SVM classifier
    params: dictionary of params to pass to contourf, optional
    """
    
    def make_meshgrid(x, y,h=0.02):
        """Create a mesh of points to plot in
    
        Parameters
        ----------
        x: data to base x-axis meshgrid on
        y: data to base y-axis meshgrid on
        h: stepsize for meshgrid, optional
    
        Returns
        -------
        xx, yy : ndarray
        """
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        return xx, yy
    
    
    X0, X1 = points[:, 0], points[:, 1]

    
    xx, yy = make_meshgrid(X0, X1,0.5)


    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    
    #VISUALIZATION OF DECISION
    fig2 = plt.figure(1,figsize=(5,5))
    ax = fig2.add_subplot(1,1,1)
    out = plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    
    ax.scatter(X0, X1, c=labels, cmap=plt.cm.coolwarm, s=80, edgecolors='k')
    
    ax.scatter(X0[clf.support_],X1[clf.support_],c=labels[clf.support_], cmap=plt.cm.coolwarm, s=80, edgecolors='w')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    
    plt.show()
    
    return out