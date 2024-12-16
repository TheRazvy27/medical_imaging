import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as sc
import scipy.ndimage.morphology as morpho

def rgb2gri(img_in, format):
    img_in = img_in.astype('float')
    s = img_in.shape
    if len(s) == 3 and s[2] == 3:
        if format == 'png':
            img_out = (0.299 * img_in[:,:,0] + 0.587 * img_in [:,:,1] + 0.114 * img_in[:,:,2])*255
        elif format == 'jpg':
            img_out = 0.299 * img_in[:, :, 0] + 0.587 * img_in[:, :, 1] + 0.114 * img_in[:, :, 2]
        img_out = np.clip(img_out ,0, 255)
        img_out = img_out.astype('uint8')
        return img_out
    else:
        print ('conversia nu a putut fi realizata deoarece imaginea de intrare nu este color ')
        return img_in

def contrast_liniar_portiuni(img_in, L, a, b, Ta, Tb):
    s = img_in.shape
    img_out = np.empty_like(img_in)
    img_in = img_in.astype(float)
    for i in range(0, s[0]):
        for j in range(0, s[1]):
            if (img_in[i, j] < a):
                img_out[i, j] = (Ta / a) * img_in[i, j]
            if (img_in[i, j] >= a and img_in[i, j] <= b):
                img_out[i, j] = Ta + ((Tb - Ta) / (b - a)) * (img_in[i, j] - a)
            if (img_in[i, j] > b):
                img_out[i, j] = Tb + ((L - 1 - Tb) / (L - 1 - b)) * (img_in[i, j] - b)

    img_out = np.clip(img_out, 0, 255)
    img_out = img_out.astype('uint8')
    return img_out

def extindere_max_contrast(img_in, L, a, b):
    img_out = contrast_liniar_portiuni(img_in, L, a, b, 0, 255)
    return img_out

def logaritmic(img_in, L):
    s = img_in.shape
    img_out = np.empty_like(img_in)
    img_in = img_in.astype(float)
    for i in range(0, s[0]):
        for j in range(0, s[1]):
                img_out[i, j] = ((L -1)/np.log(L))*np.log (img_in[i,j] +1 )

    img_out = np.clip(img_out, 0, 255)
    img_out = img_out.astype('uint8')
    return img_out

def putere(img_in, L, r, a):
    s = img_in.shape
    img_out = np.empty_like(img_in)
    img_in = img_in.astype(float)
    for i in range(0, s[0]):
        for j in range(0, s[1]):
            if (img_in[i,j] <a):
                img_out[i, j] = a*(img_in[i,j]/a)**r
            else:
                img_out[i, j] = L - 1 - (L - 1 - a)*((L - 1 -img_in[i,j]) /(L - 1 - a))**r

    img_out = np.clip(img_out, 0, 255)
    img_out = img_out.astype('uint8')
    return img_out

def binarizare (img_in ,a):
    s = img_in.shape
    img_out = np.empty_like(img_in)
    img_in = img_in.astype(float)
    for i in range(0, s[0]):
        for j in range(0, s[1]):
            if (img_in[i, j] < a):
                img_out[i,j] = 255
            else:
                img_out[i,j] = 0

    img_out = np.clip(img_out, 0, 255)
    img_out = img_out.astype('uint8')
    return img_out

cale = r'C:\Users\NicolaeRazvanStoica\Desktop\facultate\poze'

files = os.listdir(cale)
for i in files:
    cale_img = os.path.join(cale, i)
    img_plt = plt.imread(cale_img)
    img_gray_tones = rgb2gri(img_plt, 'jpg')
    img_max_contrast = extindere_max_contrast(img_gray_tones, 256, 100, 220)
    img_log = logaritmic(img_gray_tones, 256)
    img_putere = putere(img_gray_tones, 256, 4.5, 70)
    limita = np.mean(img_max_contrast) + 30
    # img_bin = binarizare(img_max_contrast, 180)
    img_bin = binarizare(img_max_contrast, limita)
    s = np.ones([3, 3], dtype='uint8')
    filtrarea = sc.binary_dilation(img_bin, structure=s)
    img_zona_interes = img_bin / 255 * img_gray_tones
    plt.subplot(5, 3, 1), plt.imshow(img_plt, cmap = 'gray'), plt.title('Original Image')
    plt.subplot(5, 3, 2), plt.imshow(img_gray_tones, cmap='gray'), plt.title('Grayscale Image')
    plt.subplot(5, 3, 3), plt.imshow(img_max_contrast, cmap='gray'), plt.title('Max Contrast Image')
    plt.subplot(5, 3, 7), plt.imshow(img_log, cmap='gray'), plt.title('Logarithmic Image')
    plt.subplot(5, 3, 8), plt.imshow(img_putere, cmap='gray'), plt.title('Fixed Point Power Image')
    plt.subplot(5, 3, 9), plt.imshow(img_bin, cmap='gray'), plt.title('Binary Image')
    dil = morpho.binary_dilation(img_bin, s)
    dill = morpho.binary_dilation(dil, s)
    # plt.subplot(5, 3, 13), plt.imshow(dil, cmap = 'gray'), plt.title('Dilated')
    plt.subplot(5, 3, 13), plt.imshow(img_zona_interes, cmap='gray'), plt.title('ROI Selection')
    plt.subplot(5, 3, 14), plt.imshow(filtrarea, cmap='gray'), plt.title('Dilated Image')
    plt.show()

