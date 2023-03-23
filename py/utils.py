import numpy as np

from scipy.interpolate import make_interp_spline
from scipy import ndimage
import scipy.optimize as op

from matplotlib import pyplot as plt

import emcee
import corner

def create_median_map(Xin, Yin, weights, xy_range, bins=50, 
                      minstars=2, gaussfilter=None):
    # Weighted 2D Histogram
    H,X,Y=np.histogram2d(Xin, Yin, weights=weights, 
                         bins=bins,range=[xy_range[:2], xy_range[2:]])
    XX,YY=np.meshgrid(X,Y)
    X1 = (X[:-1] - X[1:])/2 +X[:-1]
    Y1 = (Y[:-1] - Y[1:])/2 + Y[:-1]
    
    # Unweighted 2D Histogram
    Hc,X1,X2 = np.histogram2d(Xin, Yin, bins=bins, range=[xy_range[:2], xy_range[2:]])
    Hc[Hc<minstars]=1
    Ht = H/Hc
    
    # Gaussian filter
    if gaussfilter is not None:
        H = ndimage.gaussian_filter(Ht,sigma=gaussfilter,order=0)
    else:
        H = Ht
    # Filter out bins with less than minstars
    Hc,X,Y=np.histogram2d(Xin, Yin, bins=bins,range=[xy_range[:2], xy_range[2:]])
    H[Hc<minstars]=np.NaN
    return XX, YY, H

def running_median(X, Y, nbins=10):
    bins = np.linspace(np.nanquantile(X, 0.005),np.nanquantile(X, 0.995), nbins)
    delta = bins[1]-bins[0]
    idx  = np.digitize(X, bins, right=False)
    med = [np.nanquantile(Y[idx==k], 0.5) for k in range(1, nbins)]
    q16 = [np.nanquantile(Y[idx==k], 0.16) for k in range(1, nbins)]
    q84 = [np.nanquantile(Y[idx==k], 0.84) for k in range(1, nbins)]
    N   = np.array([len(Y[idx==k]) for k in range(1, nbins)])
    return bins[1:] - delta/2, med, q16, q84, N

# Overplot running median
def overplot_trend(X, Y, alpha=0.2, color="k", bins=10, label="", lw=3):
    #lines = mdline(X, Y, bins=bins)
    lines = running_median(X, Y, nbins=bins)
    xnew  = np.linspace(lines[0].min(), lines[0].max(), bins)
    spl   = make_interp_spline(lines[0], lines[1], k=2)
    power_smooth = spl(xnew)
    plt.plot(lines[0], lines[1], ms=50, color=color, lw=lw, label=label)
    plt.fill_between(lines[0], lines[2], lines[3], alpha=alpha, color=color)

def overplot_representative_errorbars(x, y, xerr, yerr, 
                                      age_ranges = [[0,2], [2,4], [4,8], [8,15]], 
                                      y_plot=11.5, **kwargs):
    for ii in np.arange(len(age_ranges)):
        sel = (x > age_ranges[ii][0]) & (x < age_ranges[ii][1])
        xerr_ii=np.median(xerr[sel])
        yerr_ii=np.median(yerr[sel])
        plt.errorbar(0.5*(age_ranges[ii][0]+age_ranges[ii][1]), y_plot,
                     xerr=xerr_ii, yerr=yerr_ii, **kwargs)

    
##### Utilities to fit the [Fe/H] gradient (inherited from Anders+2017b)

# Define the probability function as likelihood * prior.
def lnprior(theta, Nfit=3):
    if Nfit == 3:
        m, b, f = theta
        if -0.2 < m < 0.2 and -0.5 < b < 1.0 and 0 < f < 1.0:
            return 0.0
    if Nfit == 4:
        m, b, ms, bs = theta
        if -0.2 < m < 0.2 and -0.5 < b < 1.0 and \
           -0.9 < ms < 0.9 and -.9< bs <.9:
            return 0.0
    return -np.inf

def lnlike(theta, x, y, yerr, w, Nfit=3):
    if Nfit == 3:
        # linear slope + constant scatter
        m, b, f = theta
        model   = m * x + b
        sigma2  = ( yerr**2. + f**2. )
    elif Nfit == 4:
        # linear slope + scatter \propto R
        m, b, ms, bs = theta
        model   = m * x + b
        sigma2  = ( yerr**2. + (ms*x + bs)**2. )
    return -0.5*( np.sum((y-model)**2./sigma2 + np.log(sigma2) - 2.*np.log(w) ) )

def lnprob(theta, x, y, yerr, w, Nfit=3):
    lp = lnprior(theta, Nfit=Nfit)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr, w, Nfit=Nfit)

def fit_gradient_with_dispersion(x, y, yerr, w, Nfit=3, agebounds=[0,1], guide=False):
    
    savetext=str(agebounds[0])+"_"+str(agebounds[1])
    # Do the least-squares fit and compute the uncertainties.
    A = np.vstack((np.ones_like(x), x)).T
    C = np.diag(yerr * yerr)
    cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
    b_ls, m_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))
    print("""Least-squares results:
        m = {0} ± {1} 
        b = {2} ± {3} 
    """.format(m_ls, np.sqrt(cov[1, 1]), b_ls, np.sqrt(cov[0, 0])))

    # Find the maximum likelihood value.
    chi2 = lambda *args: -2 * lnlike(*args)
    if Nfit == 3:
        result = op.minimize(chi2, [-0.06, 0.5, 0.2], args=(x, y, yerr, w))
        m_ml, b_ml, f_ml = result["x"]
        print("""Maximum likelihood result:
            m = {0} 
            b = {1} 
            f = {2} 
        """.format(m_ml, b_ml, f_ml))
    elif Nfit == 4:
        result = op.minimize(chi2, [-0.06, 0.5, 0., 0.2],
                             args=(x, y, yerr, w))
        m_ml, b_ml, ms_ml, bs_ml = result["x"]
        print("""Maximum likelihood result:
            m  = {0} 
            b  = {1} 
            ms = {2} 
            bs = {3} 
        """.format(m_ml, b_ml, ms_ml, bs_ml))

    # Set up the sampler.
    ndim, nwalkers = Nfit, 100
    pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr, w))

    # Clear and run the production chain.
    print("Running MCMC...")
    sampler.run_mcmc(pos, 750, rstate0=np.random.get_state())
    print("Done.")

    # Make a corner plot.
    burnin = 250
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    if Nfit == 3:
        fig = corner.corner(samples, labels=[r"$m\,\, $[dex/kpc]",
                                             r"$b\,$ [dex]",
                                             r"$\sigma\,$ [dex]"],
                            show_titles=True, title_kwargs={"fontsize": 14}, 
                            title_fmt=".3f")
        fig.gca().annotate("APOGEE DR17, "+str(agebounds[0])+
                       r" Gyr $<\,\tau<\,$ "+str(agebounds[1])+" Gyr",
                       xy=(0.7, 0.85), xycoords="figure fraction",
                       xytext=(0, -5), textcoords="offset points",
                       ha="center", va="top", fontsize=16)
    elif Nfit == 4:
        fig = corner.corner(samples, labels=[r"$m\,\, $[dex/kpc]",
                                             r"$b\,$ [dex]",
                                             r"$m_{\sigma}\,$ [dex/kpc]",
                                             r"$b_{\sigma}\,$ [dex]"],
                            show_titles=True, title_kwargs={"fontsize": 14}, 
                            title_fmt=".3f")
        fig.gca().annotate("APOGEE DR17, "+str(agebounds[0])+
                       r" Gyr $<\,\tau<\,$ "+str(agebounds[1])+" Gyr",
                       xy=(0.7, 0.85), xycoords="figure fraction",
                       xytext=(0, -5), textcoords="offset points",
                       ha="center", va="top", fontsize=16)
    if guide:
        plt.savefig("../im/feh_gradient_guide_agebin" + savetext +"_corner.png")
    else:
        plt.savefig("../im/feh_gradient_agebin" + savetext +"_corner.png")
        
    # Compute the quantiles.
    if Nfit == 3:
        m_mcmc, b_mcmc, f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                     zip(*np.percentile(samples, [16, 50, 84],
                                                        axis=0)))
        print("""MCMC result:
            m = {0[0]} +{0[1]} -{0[2]} 
            b = {1[0]} +{1[1]} -{1[2]} 
            f = {2[0]} +{2[1]} -{2[2]} 
        """.format(m_mcmc, b_mcmc, f_mcmc))
    elif Nfit == 4:
        m_mcmc, b_mcmc, ms_mcmc, bs_mcmc = map(lambda v: (v[1], v[2]-v[1],
                                                          v[1]-v[0]),
                                     zip(*np.percentile(samples, [16, 50, 84],
                                                        axis=0)))
        print("""MCMC result:
            m  = {0[0]} +{0[1]} -{0[2]} 
            b  = {1[0]} +{1[1]} -{1[2]} 
            ms = {2[0]} +{2[1]} -{2[2]} 
            bs = {3[0]} +{3[1]} -{3[2]} 
        """.format(m_mcmc, b_mcmc, ms_mcmc, bs_mcmc))

    # Plot the dataset.
    plt.figure()
    plt.axis([5,11,-1,.6])
    xl = np.array([5, 11])
    #color= np.zeros((4,len(w))); color[2,:]= 1.; color[3,:]= w  # Set alpha
    #for kk in np.arange(N):
    plt.errorbar(x, y, yerr=yerr, fmt=".k", c="k", alpha=0.2)
    if guide:
        plt.xlabel(r"$R_{\rm guide}$ [kpc]", fontsize=20)
    else:
        plt.xlabel(r"$R_{\rm Gal}$ [kpc]", fontsize=20)
    plt.ylabel("[Fe/H]", fontsize=20)
    plt.tight_layout()
    # Plot the best-parameter result.
    if Nfit == 3:
        plt.fill_between(xl, m_mcmc[0]*xl+b_mcmc[0]-f_mcmc[0],
                        m_mcmc[0]*xl+b_mcmc[0]+f_mcmc[0], alpha=0.1)
    elif Nfit == 4:
        plt.fill_between(xl,(m_mcmc[0]-ms_mcmc[0])*xl+(b_mcmc[0]-bs_mcmc[0]),
                        (m_mcmc[0]+ms_mcmc[0])*xl+(b_mcmc[0]+bs_mcmc[0]),
                        alpha=0.1)
    plt.plot(xl, m_mcmc[0]*xl+b_mcmc[0], "k", c='blue')
    # Plot the least-squares result.
    plt.plot(xl, m_ls*xl+b_ls, "--k")
    # Plot the maximum likelihood result.
    plt.plot(xl, m_ml*xl+b_ml, "k", lw=2)

    # Plot some samples onto the data.
    if Nfit == 3:
        for m, b, f in samples[np.random.randint(len(samples), size=100)]:
            plt.plot(xl, m*xl+b, color="k", alpha=0.1)
    elif Nfit == 4:
        for m,b,ms,bs in samples[np.random.randint(len(samples), size=100)]:
            plt.plot(xl, m*xl+b, color="k", alpha=0.1)
    plt.errorbar(x, y, yerr=yerr, fmt=".k", c="k", alpha=0.2)
    if guide:
        plt.savefig("../im/feh_gradient_guide_agebin" + savetext +"_fit.png")
    else:
        plt.savefig("../im/feh_gradient_agebin" + savetext +"_fit.png")
    
    if Nfit == 3:
        return m_mcmc, b_mcmc, f_mcmc
    elif Nfit == 4:
        return m_mcmc, b_mcmc, ms_mcmc, bs_mcmc
        