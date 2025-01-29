
#    * patchsiz - patch size (pix)
#    * patchenv - patch spatial envelope s.d. (pix)
#    * patchlum - patch background luminance [optional]
#    * gaborper - Gabor spatial period (pix)
#    * gaborang - Gabor orientation (rad)
#    * gaborphi - Gabor unit phase [optional]
#    * gaborcon - Gabor Michelson contrast at center
#    * noisedim - noise dimension (pix) [optional]
#    * noisecon - noise RMS contrast at center



import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import signal
import numpy as np


patchsiz = 40
gaborper = 5
gaborphi = 0*np.pi/180
gaborang = 90*np.pi/180
gaborcon = 0.3
noisecon = 1
patchenv = 7
noisedim = gaborper/6


# Grating patch
x = np.arange(1,patchsiz+1)
y = np.arange(1,patchsiz+1)
xv, yv  = np.meshgrid(x-(patchsiz+1)/2,y-(patchsiz+1)/2)

r = np.sqrt(xv**2+yv**2) # radius
t = -np.arctan2(xv,yv); # anglec


u = np.sin(gaborang)*xv+np.cos(gaborang)*yv

# plt.imshow(u)

gaborimg = 0.5*np.cos(2*np.pi*(u/gaborper+gaborphi))
gaborimg = gaborimg*gaborcon
# plt.imshow(gaborimg)


# Noise patch

nfiltsiz = np.ceil(noisedim*3)*2+1
nfiltsiz = nfiltsiz.astype(int)
nfiltker = norm.pdf(np.linspace(-0.5*nfiltsiz,+0.5*nfiltsiz,nfiltsiz),0,noisedim)/norm.pdf(0,0,noisedim) # 1D filter
nfiltker = np.transpose(np.matrix(nfiltker)) * np.matrix(nfiltker) # # Now lets create the 2D gaussian filter (You must work with matrixes here)
nfiltker = nfiltker/np.sqrt(np.sum(nfiltker) ** 2)

# plt.imshow(nfiltker)

dim_mat = patchsiz + 2*nfiltsiz
noisemat = np.random.rand(dim_mat,dim_mat)

# hw to generate new figures panes
# hot to plot units in imshow
plt.imshow(noisemat)

noiseimg = signal.convolve2d(noisemat, nfiltker, mode='same', boundary='symm')
noiseimg = noiseimg[nfiltsiz:-nfiltsiz,nfiltsiz:-nfiltsiz] #match the size with grating patch
noiseimg = noiseimg * noisecon
#plt.figure(2)
noisefig = plt.imshow(noiseimg)
#cbar = colorbar(noiseimg , ticks=[-1, 0, 1], orientation='horizontal')

envelop = (norm.pdf(r,0,patchenv)/norm.pdf(0,0,patchenv))


patchimg = (gaborimg+noiseimg)*envelop


# plt.imshow(envelop)


# plt.imshow(patchimg, cmap='gray')