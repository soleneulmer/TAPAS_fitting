#!/usr/bin/python

# TAPAS telluric correction
# 03 -10 -17

import os
import time
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from PyAstronomy import pyasl
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.stats import chisquare
from astropy.convolution import Gaussian1DKernel, convolve


def read_fits_all(filename):
    # Read tac with new wavelength
    # in this case CRIRES, input wl (TAC) and lambda wl (tac)
    # are equals, because they are both in vaccum,
    # only change to angstrom to microns
    hdu = fits.open(filename)
    head = hdu[1].header
    data1 = hdu[1].data
    data2 = hdu[2].data
    data3 = hdu[3].data
    wl = [(data1.field('Wavelength'), data2.field('Wavelength'), data3.field('Wavelength'))]
    flux = [(data1.field('Extracted_OPT'), data2.field('Extracted_OPT'), data3.field('Extracted_OPT'))]
    error = [(data1.field('Error_OPT'), data2.field('Error_OPT'), data3.field('Error_OPT'))]
    return wl, flux, error, head


def extract_range(wl, flux, wl_start, wl_end):
    """
    Finds closest values to the wl_start and wl_end
    and return this range for the wl and the flux
    """
    idx_start = np.argmin(np.abs(wl_start - wl)) - 10
    idx_end = np.argmin(np.abs(wl_end - wl)) + 10
    return wl[idx_start:idx_end], flux[idx_start:idx_end]


def exclude_regions(wl, flux, boundaries):
    # Choose points closest to boundaries: wl_start and wl_end
    # Returns the wavelength and Flux outside of the boundaries
    print(boundaries)
    if boundaries is None:
        return wl, flux
    else:
        wl_start = boundaries[0]
        wl_end = boundaries[1]
        idx_start = max(0, np.argmin(np.abs(wl_start - wl)) - 10)
        idx_end = np.argmin(np.abs(wl_end - wl)) + 10
        wl_cut = np.append(wl[:idx_start], wl[idx_end:])
        flux_cut = np.append(flux[:idx_start], flux[idx_end:])
        print(idx_start, idx_end)
        return wl_cut, flux_cut


def read_tapas(file_tapas, wl, flux, rvshift=False):
    raw_wl, raw_trans = np.loadtxt(file_tapas, skiprows=38, unpack=True)
    wl_tapas = raw_wl[::-1]
    trans_tapas = raw_trans[::-1]
    if rvshift:
        print('Doppler shift TAPAS transmission')
        rv, cc = pyasl.crosscorrRV(
            wl, flux, wl_tapas, trans_tapas,
            rvmin=-60., rvmax=60.0, drv=0.1, mode='doppler', skipedge=50)
        maxind = np.argmax(cc)
        print("CCF is maximized at dRV = ", rv[maxind], " km/s")
        wlcorr_tapas = wl_tapas * (1. + rv[maxind]/299792.)
        return wlcorr_tapas, trans_tapas
    else:
        return wl_tapas, trans_tapas


def f(wave, a, b, c, d, return_wave=False):
    # transform wave with polynomial
    # interpolate telluric to new wave
    # subratct fluxes
    w0 = wave[0]
    w = wave - w0
    new_wave = wave + a + b*w + c*w**2 + d*w**3
    if return_wave:
        return new_wave
    f_tapas = interp1d(wl_tapas, trans_tapas)
    try:
        new_flux_tapas = f_tapas(new_wave)
    except ValueError:
        return np.zeros_like(wave)
    return new_flux_tapas


def continuum(wave, a, b, c, trans):
    f_tapas = interp1d(wl_tapas, trans)
    try:
        flux_tapas = f_tapas(wave)
    except ValueError:
        return np.zeros_like(wave)

    new_flux_tapas = flux_tapas * (a*wave**0 + b*wave**1)  # + c*wave**2)
    return new_flux_tapas


def scaling(transmission, t, wl, coeff):
    t_tapas_M = transmission[0]
    t_tapas_R = transmission[1]
    trans_sc = continuum(wl, *coeff, trans=t_tapas_R * t_tapas_M**t)
    # t_tapas_R * t_tapas_M**(t)
    # trans_sc = transmission**(t)
    return trans_sc


def resolution(transmission, std, extend=False):
    # g = resolution_to_kernel(wl_tapas, resolution)
    # print(std)
    g = Gaussian1DKernel(stddev=std)
    if extend:
        before = transmission[g.size/2+1:]
        after = transmission[g.size/2]
        extended = np.r_[before, transmission, after]
        convolved = convolve(extended, g)
    else:
        convolved = convolve(transmission, g)
    return convolved


def resolution_to_kernel(wl, resolution):
    # From TelFit
    central_wl = (wl[0] + wl[-1])/2.0
    xspacing = wl[1] - wl[0]
    fwhm = central_wl / resolution
    sigma = fwhm / (2.0*np.sqrt(2.0*np.log(2.0)))
    x = np.arange(0, 10*sigma, xspacing)
    gaussian = np.exp(-(x-5*sigma)**2/(2*sigma**2))
    return gaussian, x


def write_tell(wl, wl_corr, flux, flux_corr, trans_tapas, error, error_c, head, obs_name):
    wl = np.asarray(wl[0]).flatten()
    flux = np.asarray(flux[0]).flatten()
    error = np.asarray(error[0]).flatten()
    error_c = np.asarray(error_c[0]).flatten()

    tbhdu = fits.BinTableHDU.from_columns(
        [fits.Column(name='lambda', format='1D', array=wl),
         fits.Column(name='mlambda', format='1D', array=wl_corr),
         fits.Column(name='flux', format='1D', array=flux),
         fits.Column(name='cflux', format='1D', array=flux_corr),
         fits.Column(name='mtrans', format='1D', array=trans_tapas),
         fits.Column(name='inputerror', format='1D', array=error),
         fits.Column(name='correrror', format='1D', array=error)])
    tbhdu.header.extend(head)
    new_hdul = fits.HDUList()
    new_hdul.append(tbhdu)
    # new_name = obs_name + 'all_tell_TAPAS.fits'
    new_hdul.writeto(obs_name, output_verify='ignore', clobber=True)
    print('Writing FITS file ', obs_name)
    return None


def write_res(chi2s, mean_scl, filename):
    tow = str(chi2s) + ' ' + str(mean_scl)
    with open(filename, 'w') as f:
        f.write(tow)
    return None


def fit_tapas(file_science, file_tapas_M, file_tapas_R, start_wl, exclude=None, mean_scl=None):
    print('\nStarting Fit tapas\n--------------')
    if mean_scl is None:
        print('Wavelength fit starts from [-0.05, 0., 0.]')
    if mean_scl is not None:
        print('Wavelength fit starts with value from Standard star')
    # exclude = None
    wl, flux, error, head = read_fits_all(file_science)
    # Transmission ONE molecule(H2O or O2)
    wl_tapas, tapas_M = read_tapas(file_tapas_M, wl[0][0], flux[0][0])
    # Transmission of OTHER molecule and Rayleigh scattering
    _, tapas_R = read_tapas(file_tapas_R, wl[0][0], flux[0][0])
    # Transmission of the atmosphere
    trans_tapas = tapas_M * tapas_R

    coeffs_std = []
    coeffs_cont = []
    coeff_scl = []
    wl_corr = []
    flux_corr = []
    trans_tapas_corr = []
    chi2s = []
    for i in range(3):
        # Transmission global: trans_tapas
        # Fit wavelength to TAPAS
        # ------------------------
        p_wl, cov_wl = curve_fit(f, wl[0][i],
                                 flux[0][i], p0=start_wl[i], method='lm')
        print('\nWavelength parameters: ', p_wl)
        coeffs_std.append(p_wl)
        new_wl = f(wl[0][i], *p_wl, return_wave=True)  # wl star shifted
        wl_corr.append(new_wl)

        # Exclude regions
        # ---------------
        if exclude is not None:
            wl_ex, flux_ex = exclude_regions(new_wl, flux[0][i], exclude[i])
        else:
            wl_ex = new_wl
            flux_ex = flux[0][i]

        # Fit continuum to the star
        # --------------------------
        continuum1 = lambda wl, a, b, c: continuum(wl, a, b, c, trans=trans_tapas)
        # p_c, cov_c = curve_fit(continuum1, new_wl,
        #                        flux[0][i], p0=[0., 0., 0.], method='lm')
        p_c, cov_c = curve_fit(continuum1, wl_ex,
                               flux_ex, p0=[0., 0., 0.], method='lm')
        print('Continuum parameters: ', p_c)
        coeffs_cont.append(p_c)

        if mean_scl is None:
            # Fit H2O/O2 scaling to the star
            # ------------------------------
            scaling1 = lambda transmission, t: scaling(transmission, t, wl=new_wl, coeff=p_c)
            p_sc, cov_sc = curve_fit(scaling1, np.array((tapas_M, tapas_R)),
                                     flux[0][i], p0=[1.], method='lm')
            print('Scaling factor for ONE moelcule transmission: ', p_sc)
            coeff_scl.append(p_sc)
            trans_tapas_sc = scaling1(np.array((tapas_M, tapas_R)), p_sc)
            # trans_tpas_msc is wavelength ok, continuum ok, scaled ok

            # trans_tapas_corr.append(trans_tapas_sc)
            chi2 = np.sum((flux[0][i]-trans_tapas_sc)**2)
            chi2s.append(chi2)
            print('Chisquare: ', chi2)
            # print('Scipy Chi2: ', chisquare(flux[0][i], f_exp=trans_tapas_sc))
            # (f_obs - f_exp)**2 / f_exp

            if i == 2:
                # Ponderate the H2O/O2 value
                coeff_scl = np.array(coeff_scl).flatten()
                chi2s = np.array(chi2s)
                final_scaling = np.sum(coeff_scl*(1./chi2s))/np.sum((1./chi2s))
                print('\nFinal scaling factor: ', final_scaling, '\n')

                return coeffs_std, final_scaling

        if mean_scl is not None:
            # Fit H2O/O2 scaling to the star
            # ------------------------------
            scaling1 = lambda transmission, t: scaling(transmission, t, wl=new_wl, coeff=p_c)
            print('Mean Scaling factor: ', mean_scl)
            trans_tapas_sc = scaling1(np.array((tapas_M, tapas_R)), mean_scl)
            trans_tapas_corr.append(trans_tapas_sc)

            # Telluric correction
            flux_corr.append(flux[0][i]/(trans_tapas_sc))

        # # Fit resolution
        # p_res = curve_fit(resolution, trans_tapas_sc, flux[0][i], p0=[1.], method='lm')
        # print('Resolution', p_res)
        # convolved = resolution(trans_tapas_sc, p_res)
        # print(trans_tapas_sc.shape, convolved.shape)

        # Telluric correction
        # flux_corr.append(flux[0][i]/(trans_tapas_sc))

    wls_corr = np.asarray(wl_corr).flatten()
    fluxes_corr = np.asarray(flux_corr).flatten()
    trans_tapases_corr = np.asarray(trans_tapas_corr).flatten()

    red_chi2 = np.sum((np.asarray(flux[0]).flatten() - trans_tapases_corr)**2)  # 7 degrees of freedom (4 wl, 2 cont,1 scaling )
    print('\nFINAL chi2: ', red_chi2, '\n')
    # PLOTTING
    ax = plt.subplot(211)
    # plt.plot(np.asarray(wl[0]).flatten(), np.asarray(flux[0]).flatten(), 'k-', label='RAW CRIRES')
    # plt.plot(wl_tapas, trans_tapas, 'm-', label='RAW TAPAS')
    plt.plot(wls_corr, np.asarray(flux[0]).flatten(), 'C0-', label='CRIRES data, wavelength shifted')
    plt.plot(wls_corr, trans_tapases_corr, 'C3-', label='TAPAS continuum adjusted and scaled')
    plt.plot(wls_corr, fluxes_corr, 'C2-', label='Telluric corrected')
    plt.legend()
    plt.subplot(212, sharex=ax)
    plt.plot(wls_corr, np.asarray(flux[0]).flatten() - trans_tapases_corr, label='Residuals data-tapas')
    plt.plot(wls_corr, wls_corr*0., 'k-', alpha=0.5)
    plt.plot(wls_corr, wls_corr*0.+0.05, 'k--', alpha=0.5)
    plt.plot(wls_corr, wls_corr*0.-0.05, 'k--', alpha=0.5, label='5% limit')
    plt.legend()

    return wls_corr, fluxes_corr, trans_tapases_corr, float(red_chi2), float(mean_scl)


if __name__ == "__main__":

    order_nb = 48

    # Choosing the directories and files
    if order_nb == 45:
        direc_tapas = "/home/solene/atmos/tapas_janis/transmission_tapas/"
        direc_out = '/home/solene/atmos/tapas_janis/output/'
        sci_calib = direc_tapas + "sci_calib_tapas%i.dat" % (order_nb)
        sciences, calibs, tapases_M, tapases_R = np.genfromtxt(sci_calib, dtype='str')
        direc_crires = '/home/solene/atmos/janiscrires/crires_expert/order45expert/'
        print(sci_calib, tapases_M)
    elif order_nb == 48:
        direc_tapas = "/home/solene/atmos/tapas_janis/transmission_tapas/"
        direc_out = '/home/solene/atmos/tapas_janis/output/'
        sci_calib = direc_tapas + "sci_calib_tapas%i.dat" % (order_nb)
        sciences, calibs, tapases_M, tapases_R = np.genfromtxt(sci_calib, dtype='str')
        direc_crires = '/home/solene/atmos/janiscrires/crires_expert/order48expert/'

    # Fit each Observation
    for OB_nb in range(1, 4):
        start_time = time.time()
        # Input files
        file_science = direc_crires + sciences[OB_nb-1]
        file_calib = direc_crires + calibs[OB_nb-1]
        file_tapas_M = direc_tapas + tapases_M[OB_nb-1]
        file_tapas_R = direc_tapas + tapases_R[OB_nb-1]
        # Result files
        file_calib2 = os.path.splitext(file_calib)[0] + '_tapas.fits'
        result_calib = direc_out + os.path.basename(file_calib2)
        file_science2 = os.path.splitext(file_science)[0] + '_tapas.fits'
        file_res = os.path.splitext(file_science)[0] + '_tapas.res'
        result_science = direc_out + os.path.basename(file_science2)
        result_science_res = direc_out + os.path.basename(file_res)

        # TAPAS file
        wl, flux, error, head = read_fits_all(file_calib)
        print(file_tapas_M, file_tapas_R)
        wl_tapas, tapas_M = read_tapas(file_tapas_M, wl[0][0], flux[0][0])
        wl_tapas, tapas_R = read_tapas(file_tapas_R, wl[0][0], flux[0][0])
        trans_tapas = tapas_M * tapas_R

        # Standard star
        print('Standard star\n', file_calib)
        print('TAPAS files\n', file_tapas_M, '\n', file_tapas_R)
        plt.figure()
        if order_nb == 48:
            # WL coefficient starting values
            start_wl = [[-0.08, 0., 0., 0.], [-0.08, 0., 0., 0.], [-0.08, 0., 0., 0.]]
            # Stellar lines exlucsion regions
            exclude = [[1167.82, 1168.27], None, None]
        elif order_nb == 45:
            start_wl = [[-0.05, 0., 0., 0.], [-0.05, 0., 0., 0.], [-0.05, 0., 0., 0.]]
            exclude = [[1252.1, 1252.9], None, None]
        # Fitting 
        std_wl, final_scaling = fit_tapas(file_calib,
                                          file_tapas_M, file_tapas_R, start_wl, exclude)
        wl_corr, flux_corr, trans_tapas_corr, chi2s, mean_scl = fit_tapas(file_calib,
                                                                          file_tapas_M, file_tapas_R,
                                                                          start_wl, exclude, mean_scl=final_scaling)
        # Saving file
        # write_tell(np.asarray(wl[0]), wl_corr,
        #            np.asarray(flux[0]), flux_corr,
        #            trans_tapas_corr, np.asarray(error[0]),
        #            np.asarray(error[0]), head, result_calib)
        write_tell(wl, wl_corr,
                   flux, flux_corr,
                   trans_tapas_corr, error,
                   error, head, result_calib)

        # Target star
        print('\n\nTarget star\n', file_science)
        print('TAPAS files\n', file_tapas_M, '\n', file_tapas_R)
        wl, flux, error, head = read_fits_all(file_science)
        print(len(np.asarray(wl[0]).flatten()))
        plt.figure()
        # Exclusion regions
        if order_nb == 48:
            exclude = [[1168.5, 1170.7], [1176.8, 1178.1], None]
        elif order_nb == 45:
            exclude = [[1252.1, 1252.9], None, None]
        # Fitting
        science_wl, final_scaling = fit_tapas(file_science,
                                              file_tapas_M, file_tapas_R, std_wl, exclude)
        wl_corr, flux_corr, trans_tapas_corr, chi2s, mean_scl = fit_tapas(file_science,
                                                                          file_tapas_M, file_tapas_R,
                                                                          std_wl, exclude, mean_scl=final_scaling)
        print(len(np.asarray(wl[0]).flatten()), len(wl_corr))

        # # Saving file
        write_tell(wl, wl_corr,
                   flux, flux_corr,
                   trans_tapas_corr, error,
                   error, head, result_science)

        write_res(chi2s, mean_scl, result_science_res)
