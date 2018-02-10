# OpenCV-GPU-matchTemplate

This project has been created by Vito Walter Anelli and Alessandro Pagliara.

The OpenCV matchTemplate primitive for gpu has been adapted in order to enable calculations on 32-bit floating-point raw data.

The detection of the offsets that maximize the crosscorrelation between master and slave oversampled patches is a time-consuming task that typically must be repeated thousands times for each interferometric acquisition pair: it has been hence selected in this study as a good starting point for providing a preliminary assessment of GPU performances with respect to single-thread or multi-thread CPU implementations.

OpenCV libraries ver.2.4.9 are used for oversampling patches and cross-matching them in both CPU or GPU environment. The normalized cross-correlation is computed in the frequency domain (FFT) with the matchTemplate primitive, whose source code has been adapted in order to enable calculations on 32-bit floating-point raw data.
Concerning the GPU implementation, all GPU cores are involved in the oversampling and matching of a single pair of master and slave patches; the process is sequentially iterated for all the selected patches in input.


