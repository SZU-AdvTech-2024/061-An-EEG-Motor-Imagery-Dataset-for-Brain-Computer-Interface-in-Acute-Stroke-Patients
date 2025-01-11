from scipy.signal import iirnotch, lfilter

def notch_filter(X, Fs=250, Fa=50):
    """
    This function applies a notch filter to remove a specific frequency (e.g., 50 Hz) from EEG data.
    
    Parameters:
    X : numpy.ndarray
        Raw EEG data with shape [Nchans, Ntime], where Nchans is the number of channels 
        and Ntime is the number of time samples.
    Fs : float, optional
        Sampling frequency in Hz. Default is 250 Hz.
    Fa : float, optional
        Notch target frequency in Hz. Default is 50 Hz.
    
    Returns:
    Y : numpy.ndarray
        Notch-filtered data with the same shape as input X.
    """
    # Quality factor for the notch filter
    Q = 6.0

    # Calculate the notch filter parameters
    Wo = Fa / (Fs / 2)  # Normalized frequency
    BW = Wo / Q         # Bandwidth of the notch

    # Design notch filter
    b, a = iirnotch(Wo, Q)

    # Apply the notch filter along the time axis with zero phase
    Y = lfilter(b, a, X, axis=1)

    return Y
