from __future__ import division, print_function
import __builtin__
import sys
# Toggleable printing
print = __builtin__.print
print_ = sys.stdout.write
def print_on():
    global print, print_
    print =  __builtin__.print
    print_ = sys.stdout.write
def print_off():
    global print, print_
    def print(*args, **kwargs): pass
    def print_(*args, **kwargs): pass
# Dynamic module reloading
def reload_module():
    import imp, sys
    print('[cc2] reloading '+__name__)
    imp.reload(sys.modules[__name__])
def rrr():
    reload_module()
from PIL import Image
from Parallelize import parallel_compute
from Printable import DynStruct
import DataStructures as ds
import helpers
import algos
import draw_func2 as df2
import load_data2 as ld2
import numpy as np
import os, sys
import scipy.signal
import scipy.ndimage.filters as filters
from _tpl.other import imtools

import warnings

import skimage
import skimage.morphology
import skimage.filter.rank
import skimage.exposure
import skimage.util

import segmentation

# =======================================
# Parallelizable Work Functions          
# =======================================
def __compute_chip(img_path, chip_path, roi, new_size):
    '''Crops chip from image ; Converts to grayscale ; 
    Resizes to standard size ; Equalizes the histogram
    Saves as png'''
    # Read image
    img = Image.open(img_path)
    (x1, y1, x2, y2) = algos.xywh_to_tlbr(roi, img.size)
    # http://docs.wand-py.org/en/0.3.3/guide/resizecrop.html#crop-images
    # Crop out ROI: left, upper, right, lower
    #img.transform(resize='x100') #img.transform(resize='640x480>')
    raw_chip = img.crop((x1, y1, x2, y2))
    # Scale chip, but do not rotate
    chip = raw_chip.convert('L').resize(new_size, Image.ANTIALIAS)
    # Save chip to disk
    return chip

def rotate_chip(chip_path, rchip_path, theta):
    ''' reads chip, rotates, and saves'''
    chip = Image.open(chip_path)
    degrees = theta * 180. / np.pi
    rchip = chip.rotate(degrees, resample=Image.BICUBIC, expand=1)
    rchip.save(rchip_path, 'PNG')

# Why doesn't this work? 
def make_compute_chip_func(preproc_func_list):
    def custom_compute_chip(img_path, chip_path, roi, new_size):
        chip = __compute_chip(img_path, chip_path, roi, new_size)
        for preproc_func in iter(preproc_func_list):
            print('[cc2] '+preproc_func.__name__)
            chip = preproc_func(chip)
        chip.save(chip_path, 'PNG')
        return True
    return custom_compute_chip

def compute_grabcut_chip(img_path, chip_path, roi, new_size):
    seg_chip, img_mask = segmentation.segment(img_path, roi, new_size)
    raw_chip = Image.fromarray(seg_chip)
    # Scale chip, but do not rotate
    chip = raw_chip.convert('L').resize(new_size, Image.ANTIALIAS)
    chip.save(chip_path, 'PNG')
    return True

def compute_bare_chip(img_path, chip_path, roi, new_size):
    chip = __compute_chip(img_path, chip_path, roi, new_size)
    chip.save(chip_path, 'PNG')
    return True

def compute_otsu_thresh_chip(img_path, chip_path, roi, new_size):
    chip = __compute_chip(img_path, chip_path, roi, new_size)
    chip = region_normalize_chip(chip)
    chip = histeq(chip)
    chip.save(chip_path, 'PNG')
    return True

def compute_reg_norm_and_histeq_chip(img_path, chip_path, roi, new_size):
    chip = __compute_chip(img_path, chip_path, roi, new_size)
    chip = region_normalize_chip(chip)
    chip = histeq(chip)
    chip.save(chip_path, 'PNG')
    return True

def compute_reg_norm_chip(img_path, chip_path, roi, new_size):
    chip = __compute_chip(img_path, chip_path, roi, new_size)
    chip = region_normalize_chip(chip)
    chip.save(chip_path, 'PNG')
    return True

def compute_histeq_chip(img_path, chip_path, roi, new_size):
    chip = __compute_chip(img_path, chip_path, roi, new_size)
    chip = histeq(chip)
    chip.save(chip_path, 'PNG')
    return True

def compute_contrast_stretch_chip(img_path, chip_path, roi, new_size):
    chip = __compute_chip(img_path, chip_path, roi, new_size)
    chip = contrast_strech(chip)
    chip.save(chip_path, 'PNG')
    return True

def compute_localeq_contr_chip(img_path, chip_path, roi, new_size):
    chip = __compute_chip(img_path, chip_path, roi, new_size)
    chip = local_equalize(chip)
    chip.save(chip_path, 'PNG')
    return True

def compute_localeq_chip(img_path, chip_path, roi, new_size):
    chip = __compute_chip(img_path, chip_path, roi, new_size)
    chip = local_equalize(chip)
    chip = contrast_strech(chip)
    chip.save(chip_path, 'PNG')
    return True

def compute_rankeq_chip(img_path, chip_path, roi, new_size):
    chip = __compute_chip(img_path, chip_path, roi, new_size)
    chip = rank_equalize(chip)
    chip.save(chip_path, 'PNG')
    return True

# ---------------
# Preprocessing algos

def chip_decorator(func):
    def wrapper(*arg, **kwargs):
        return func(*arg, **kwargs)
    wrapper.__name__ = 'chip_decorator_'+func.__name__

def contrast_strech(chip):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chip_ = pil2_float_img(chip)
        p2 = np.percentile(chip_, 2)
        p98 = np.percentile(chip_, 98)
        chip_ = skimage.exposure.equalize_hist(chip_)
        retchip = Image.fromarray(skimage.util.img_as_ubyte(chip_)).convert('L')
    return retchip

def local_equalize(chip):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chip_ = skimage.util.img_as_uint(chip)
        chip_ = skimage.exposure.equalize_adapthist(chip_, clip_limit=0.03)
        retchip = Image.fromarray(skimage.util.img_as_ubyte(chip_)).convert('L')
    return retchip

def rank_equalize(chip):
    #chip_ = skimage.util.img_as_ubyte(chip)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chip_ = pil2_float_img(chip)
        selem = skimage.morphology.disk(30)
        chip_ = skimage.filter.rank.equalize(chip_, selem=selem)
        retchip = Image.fromarray(skimage.util.img_as_ubyte(chip_)).convert('L')
        return retchip

def skimage_historam_equalize(chip):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chip_ = pil2_float_img(chip)
        p2 = np.percentile(chip_, 2)
        p98 = np.percentile(chip_, 98)
        chip_ = skimage.exposure.rescale_intensity(chip_, in_range=(p2, p98))
        retchip = Image.fromarray(skimage.util.img_as_ubyte(chip_)).convert('L')
    return retchip


def histeq(pil_img):
    img = np.asarray(pil_img)
    return Image.fromarray(imtools.histeq(img)).convert('L')

def region_normalize_chip(chip):
    #chip = hs.get_chip(1)
    chip_ = np.asarray(chip, dtype=np.float)
    chipw, chiph = chip_.shape
    half_w = chipw * .1
    half_h = chiph * .1
    x1 = chipw/2 - half_w
    y1 = chiph/2 - half_h
    x2 = chipw/2 + half_w
    y2 = chiph/2 + half_h
    (x1,y1,x2,y2) = map(int, map(round, (x1,y1,x2,y2)))
    area = chip_[x1:x2, y1:y2]
    intensity = area.flatten()
    freq, _  = np.histogram(intensity, 32)
    #algos.viz_localmax(freq)
    maxpos  = algos.localmax(freq)
    min_int = intensity.min()
    max_int = intensity.max()
    maxima = min_int + (max_int - min_int) * np.array(maxpos) / float(len(freq))
    if len(maxima) > 2:
        low  = float(maxima[0])
        high = float(maxima[-1])
    else:
        low  = float(min_int)
        high = float(max_int)
    chip_ = (chip_ - low) * 255.0 / (high - low)
    chip_ = chip_.round()
    chip_[chip_ < 0] = 0
    chip_[chip_ > 255] = 255 
    retchip = Image.fromarray(chip_).convert('L')
    return retchip

def pil2_float_img(chip):
    return skimage.util.img_as_float(chip)
    #chip_ = np.asarray(chip, dtype=np.float)
    #if chip_.max() > 1:
        #chip_ /= 255.0
    return chip_

# =======================================
# Main Script 
# =======================================

class HotspotterChipPaths(DynStruct):
    def __init__(self):
        super(HotspotterChipPaths, self).__init__()
        self.cx2_chip_path  = []
        self.cx2_rchip_path = []
        self.chip_uid = ''

def load_chips(hs, chip_cfg=None, cx_list=None, **kwargs):
    print('\n=============================')
    print('[cc2] Precomputing chips and loading chip paths: %r' % hs.db_name())
    print('=============================')
    if chip_cfg is None:
        chip_cfg = ds.ChipConfig(**kwargs)
    else:
        chip_cfg.update(**kwargs)

    # Get or create chip paths object
    hs_cpaths = HotspotterChipPaths() if hs.cpaths is None else hs.cpaths

    if cx_list == []: return # Hack

    
    img_dir      = hs.dirs.img_dir
    rchip_dir    = hs.dirs.rchip_dir
    chip_dir     = hs.dirs.chip_dir

    cx2_gx       = hs.tables.cx2_gx
    cx2_cid      = hs.tables.cx2_cid
    cx2_theta    = hs.tables.cx2_theta
    cx2_roi      = hs.tables.cx2_roi
    gx2_gname    = hs.tables.gx2_gname

    # Get parameters
    sqrt_area   = chip_cfg.chip_sqrt_area
    grabcut     = chip_cfg.grabcut        
    histeq      = chip_cfg.histeq         
    region_norm = chip_cfg.region_norm    
    rankeq      = chip_cfg.rank_eq        
    localeq     = chip_cfg.local_eq       
    maxcontr    = chip_cfg.maxcontrast    

    chip_uid = chip_cfg.get_uid()
    print('[cc2] chip_uid = %r' % chip_uid)
    #print(helpers.indent(str(chip_cfg), '[cc2] '))

    # Full image path
    cx2_img_path = [img_dir+'/'+gx2_gname[gx] for gx in cx2_gx]

    # Paths to chip, rotated chip
    chip_format     =  chip_dir+'/CID_%d'+chip_uid+'.png'
    cx2_chip_path   = [chip_format  % cid for cid in cx2_cid]

    # Compute normalized chip sizes
    cx2_imgchip_sz = [(float(w), float(h)) for (x,y,w,h) in cx2_roi]
    if not (sqrt_area is None or sqrt_area <= 0):
        target_area = sqrt_area ** 2
        def _resz(w, h):
            ht = np.sqrt(target_area * h / w)
            wt = w * ht / h
            return (int(round(wt)), int(round(ht)))
        cx2_chip_sz = [_resz(float(w), float(h)) for (x,y,w,h) in cx2_roi]
    else: # no rescaling
        cx2_chip_sz = [(int(w), int(h)) for (x,y,w,h) in cx2_roi]

    # --- COMPUTE CHIPS --- # 
    pcc_kwargs = {
        'arg_list'  : [cx2_img_path, cx2_chip_path, cx2_roi, cx2_chip_sz],
        'lazy'      : not hs.args.nocache_chips,
        'num_procs' : hs.args.num_procs }

    if grabcut:
        parallel_compute(compute_grabcut_chip, **pcc_kwargs)
    elif region_norm and histeq: 
        parallel_compute(compute_reg_norm_and_histeq_chip, **pcc_kwargs)
    elif region_norm: 
        parallel_compute(compute_reg_norm_chip, **pcc_kwargs)
    elif histeq: 
        parallel_compute(compute_histeq_chip, **pcc_kwargs)
    elif rankeq:
        parallel_compute(compute_rankeq_chip, **pcc_kwargs)
    elif localeq and maxcontr:
        parallel_compute(compute_localeq_contr_chip, **pcc_kwargs)
    elif localeq:
        parallel_compute(compute_localeq_chip, **pcc_kwargs)
    elif maxcontr:
        parallel_compute(compute_contrast_stretch_chip, **pcc_kwargs)
    else:
        parallel_compute(compute_bare_chip, **pcc_kwargs)

    # --- ROTATE CHIPS --- # 
    #cx2_rchip_path  = [rchip_format % cid for cid in cx2_cid]
    rchip_format = rchip_dir + '/CID_%d' + chip_uid + '.rot.png'
    cx2_rchip_path = [rchip_format % cid if cx2_theta[cx] != 0 else cx2_chip_path[cx]
                      for (cx, cid) in enumerate(cx2_cid)]
    pcc_kwargs['arg_list'] = [cx2_chip_path, cx2_rchip_path, cx2_theta] 
    parallel_compute(rotate_chip, **pcc_kwargs)

    # --- RETURN CHIP PATHS --- #

    print('[cc2] Done Precomputing chips and loading chip paths')

    # Build hotspotter path object
    hs_cpaths.cx2_chip_path  = cx2_chip_path
    hs_cpaths.cx2_rchip_path = cx2_rchip_path
    hs_cpaths.chip_uid = chip_cfg.get_uid()
    return hs_cpaths
    
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    # --- LOAD DATA --- #
    db_dir = ld2.DEFAULT
    db_dir = ld2.WS_HARD
    hs = ld2.HotSpotter()
    hs.load_tables(db_dir)
    hs.set_samples()
    # --- LOAD CHIPS --- #
    hs_cpaths = load_chips(hs)
    hs.cpaths = hs_cpaths
    cx = helpers.get_arg_after('--cx', type_=int)
    if not cx is None:
        df2.show_chip(hs, cx, draw_kpts=False)
    else:
        print('usage: feature_compute.py --cx [cx]')
    exec(df2.present())
