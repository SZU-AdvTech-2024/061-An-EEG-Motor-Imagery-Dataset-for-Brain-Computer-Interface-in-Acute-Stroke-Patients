from scipy.signal import butter, firwin, lfilter

def bandpass_filter(X, Fs=250, Fl=0.5, Fh=50, N=None, filter_type='but'):
    """
    Bandpass Filter function for EEG data.
    
    Parameters:
    X : numpy.ndarray
        Raw EEG data with shape [Nchans, Ntime], where Nchans is the number of channels 
        and Ntime is the number of time samples.
    Fs : float, optional
        Sampling frequency in Hz. Default is 250 Hz.
    Fl : float, optional
        Low cutoff frequency in Hz. Default is 0.5 Hz.
    Fh : float, optional
        High cutoff frequency in Hz. Default is 50 Hz.
    N : int, optional
        Filter order. Default is 4 for IIR (Butterworth) and 25 for FIR.
    filter_type : str, optional
        Filter type, 'but' for Butterworth (default) and 'fir' for FIR.
        
    Returns:
    Y : numpy.ndarray
        Filtered data with the same shape as input X.
    """
    # Nyquist frequency
    Fn = Fs / 2

    # Set filter order defaults based on filter type
    if N is None:
        N = 2 if filter_type == 'but' else 25

    # Compute filter coefficients based on filter type
    if filter_type == 'but':
        # Design Butterworth bandpass filter
        if (Fl + Fh) / 2 == 31.2:
            # Notch filter if the center frequency is 31.2 Hz (as per your MATLAB condition)
            B, A = butter(N, [Fl / Fn, Fh / Fn], btype='bandstop')
        else:
            B, A = butter(N, [Fl / Fn, Fh / Fn], btype='bandpass')
            
    elif filter_type == 'fir':
        # Design FIR bandpass filter
        B = firwin(N + 1, [Fl / Fn, Fh / Fn], pass_zero=False)
        A = 1.0  # FIR filter in scipy uses only B coefficients

    # Apply the filter to the data
    Y = lfilter(B, A, X, axis=1)

    return Y
