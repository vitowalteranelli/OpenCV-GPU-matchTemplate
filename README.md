# OpenCV-GPU-matchTemplate

This project has been created by Vito Walter Anelli and Alessandro Pagliara.

The OpenCV matchTemplate primitive for gpu has been adapted in order to enable calculations on 32-bit floating-point raw data.

The detection of the offsets that maximize the crosscorrelation between master and slave oversampled patches is a time-consuming task that typically must be repeated thousands times for each interferometric acquisition pair: it has been hence selected in this study as a good starting point for providing a preliminary assessment of GPU performances with respect to single-thread or multi-thread CPU implementations.

OpenCV libraries ver.2.4.9 are used for oversampling patches and cross-matching them in both CPU or GPU environment. The normalized cross-correlation is computed in the frequency domain (FFT) with the matchTemplate primitive, whose source code has been adapted in order to enable calculations on 32-bit floating-point raw data.
Concerning the GPU implementation, all GPU cores are involved in the oversampling and matching of a single pair of master and slave patches; the process is sequentially iterated for all the selected patches in input.

If you are interested in it please contact us:

Vito Walter Anelli - vitowalter.anelli@poliba.it  
[Research Lab page](http://sisinflab.poliba.it/anelli/)  
[linkedin](https://www.linkedin.com/in/vito-walter-anelli-98a9b375/)  

Alessandro Pagliara   
[linkedin](https://www.linkedin.com/in/alessandro-pagliara-b3222896/)  

Feel free to use or extend our code, just cite this:

Efficient implementation of InSAR time-consuming algorithm kernels on GPU environment

@inproceedings{guerriero2015efficient,  
  title={Efficient implementation of InSAR time-consuming algorithm kernels on GPU environment},  
  author={Guerriero, Andrea and Anelli, Vito Walter and Pagliara, Alessandro and Nutricato, Raffaele and Nitti, Davide Oscar},  
  booktitle={Geoscience and Remote Sensing Symposium (IGARSS), 2015 IEEE International},  
  pages={4264--4267},  
  year={2015},  
  organization={IEEE}  
}  
