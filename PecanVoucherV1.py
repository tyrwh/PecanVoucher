from plantcv import plantcv as pcv
import numpy as np
import pandas as pd
import multiprocessing as mp
import cv2
import re
import pytesseract
import argparse
from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import KMeans
from skimage import filters
from Levenshtein import distance as lev

def options():
    parser = argparse.ArgumentParser(description='Cranberry external image processing V1.')
    parser.add_argument('-i', '--input', help='Input file(s). Must be valid image or directory with valid images.', required=True)
    parser.add_argument('-o','--output', help='Output CSV file. File name will be created if not specified.')
    parser.add_argument('-s','--size', help='Diameter in cm of size marker. If none provided, sizes will be reported in pixels')
    parser.add_argument('-w','--writeimg', help='Write out annotated images.', default=False, action='store_true')
    parser.add_argument('-a','--annotated', help='Directory for annotated image files. If -w flag on and no path specified, one will be created.')
    parser.add_argument('-n', '--nproc', help='Number of processes to use. Cannot exceed number of CPUs', default=1, type=int)
    parser.add_argument('--blue', help='Use blue-background mode', default=False, action='store_true')
    args = parser.parse_args()
    return args

def valid_img(Path):
    return Path.suffix in ['.jpg','.jpeg','.tif','.tiff','.png','.CR2']

def clean_args(args):
    # main() was getting crowded, moved arg validation to separate fxn for legibility
    # check that the input is a valid image or contains a valid image
    if not args.input:
        raise Exception('Must specify input directory or images') 
    else:
        args.input = Path(args.input)
        if not args.input.exists():
            raise Exception('Specified input does not exist: %s' % args.input)
        else:
            output_stem = args.input.stem
            if args.input.is_dir():
                subfiles = args.input.glob('*')
                if not any([valid_img(x) for x in subfiles]):
                    raise Exception('No valid images (.jpg, .jpeg, .png, .tif, .tiff) found in specified directory:\n%s' % args.input)
            else:
                if not valid_img(args.input):
                    raise Exception('Specified img is not a valid image (.jpg, .jpeg, .png, .tif, .tiff):\n%s' % args.input)
    # validate the output filepath and create one if none provided
    # TODO (potential)- request user input if overwriting CSV? I find that annoying, but could avoid overwrite issues
    if args.output:
        args.output = Path(args.output)
        if not args.output.suffix == '.csv':
            raise Exception('Specified output file is not a valid CSV:\n%s' % args.output)
        if not args.output.parent.exists():
            raise Exception('Specified output file is in a nonexistent directory:\n%s' % args.output)
    else:
        args.output = Path(output_stem + '_ExtResults').with_suffix('.csv')
    # if annotated dir provided, make it (if none exists) and validate it
    # if none provided but writeimg flagged, make a path first and then do the same
    if args.annotated:
        args.annotated = Path(args.annotated)
        if not args.annotated.exists():
            args.annotated.mkdir(exist_ok=True)
        if not args.annotated.is_dir():
            raise Exception('Specified annotated img dir is not a valid directory:/n%s' % args.annotated)
    elif args.writeimg:
        args.annotated = Path(output_stem + '_annotated')
        args.annotated.mkdir(exist_ok=True)
    if args.nproc > mp.cpu_count():
        raise Exception('Number of processes cannot exceed number available CPUs.\nNumber processes: %s\nNumber CPUs: %s' % (args.nproc, mp.cpu_count()))
    return args

def orientedBoundingBox(contr):
    # from a cv2-generated contour, rotate and create an oriented bounding box in same format as tuple created by cv2.minAreaRect()
    # could expedite by integrating rotateContour() below, but nice that this is standalone
    (center_x,center_y),(MA,ma),angle = cv2.fitEllipse(contr)
    xmin = np.min(contr[:,:,0])
    ymin = np.min(contr[:,:,1])
    w = np.max(contr[:,:,0]) - xmin
    h = np.max(contr[:,:,1]) - ymin
    # make a canvas large enough to fit any arbitrary contour/rotated contour, redraw the contour on it, then rotate
    # draw it such that the center of the ellipse-of-best-fit is exactly at the midpoint
    drawn_contr = np.zeros((h+w,h+w), dtype = np.uint8)
    cv2.drawContours(drawn_contr, [contr - [int(center_x),int(center_y)] + [int((w+h)/2),int((h+w)/2)]], 0, 255, cv2.FILLED)
    rot_matrix = cv2.getRotationMatrix2D(((h+w)/2, (h+w)/2), angle, 1)
    rotated = cv2.warpAffine(drawn_contr, rot_matrix, (h+w,h+w))
    # redraw contours in order to find oriented bounding box (OBB)
    contr_rotated, hierarchy = cv2.findContours(rotated, 0, 2)
    obb_x,obb_y,obb_w,obb_h = cv2.boundingRect(contr_rotated[0])
    # offset vector between oriented rectangle center and ellipse center
    # tends to be small, but makes the annotations cleaner
    center_offset = np.array((obb_x + (obb_w/2) - (w+h)/2, obb_y + (obb_h/2) - (w+h)/2))
    center_offset = np.matmul(center_offset, cv2.getRotationMatrix2D((0,0), angle, 1)[:,0:2])
    rect_center_x = center_x + center_offset[0]
    rect_center_y = center_y + center_offset[1]
    obb = ((rect_center_x, rect_center_y), (obb_w, obb_h), angle)
    return obb

def rotateContour(contr):
    # from a cv2-generated contour, rotate it to be oriented vertically (long axis pointed up and down)
    # note: produces a rotated MASK, not a contour
    (center_x,center_y),(MA,ma),angle = cv2.fitEllipse(contr)
    xmin = np.min(contr[:,:,0])
    ymin = np.min(contr[:,:,1])
    w = np.max(contr[:,:,0]) - xmin
    h = np.max(contr[:,:,1]) - ymin
    drawn_contr = np.zeros((h+w,h+w), dtype = np.uint8)
    cv2.drawContours(drawn_contr, [contr - [xmin,ymin] + [int(h/2),int(w/2)]], 0, 255, cv2.FILLED)
    rot_matrix = cv2.getRotationMatrix2D((int((w+h)/2), int((w+h)/2)), angle, 1)
    rotated = cv2.warpAffine(drawn_contr, rot_matrix, (h+w,h+w))
    return rotated

def roundnessContour(contr):
    # ISO definition of roundness
    mask = np.zeros((np.max(contr)*2, np.max(contr)*2), dtype=np.uint8)
    _, radius_lrg = cv2.minEnclosingCircle(contr)
    cv2.drawContours(mask, [contr], 0, 255, cv2.FILLED)
    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, radius_sml, _, center = cv2.minMaxLoc(dist_map)
    return radius_sml / radius_lrg

def roundnessMask(mask):
    # deprecated form of roundness function, keep in for now
    pts, hierarchy = cv2.findContours(mask, 0, 2)
    _, radius_lrg = cv2.minEnclosingCircle(pts[0])
    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, radius_sml, _, center = cv2.minMaxLoc(dist_map)
    return (radius_sml-1) / radius_lrg

def ovoid(rotated):
    # approximate the ovoid metric from Tomato Analyzer
    # requires a rotated binary mask
    rowsums = np.sum(rotated, 1)
    W = np.max(rowsums/255)
    Wloc = np.argmax(rowsums)
    sumtop = np.sum(rotated[0:Wloc,:])/255
    sumbot = np.sum(rotated[Wloc:rotated.shape[0],:])/255
    y1 = np.where(rowsums > 0)[0][0]
    y2 = np.where(rowsums > 0)[0][-1]
    w1 = sumtop / (Wloc - y1)
    w2 = sumbot / (y2 - Wloc)
    ovo = 0.5 * (1 - w2/W + w1/W)
    # correct form is 0.5 * f(Wloc - y2) * (1 - w2/W + w1/W) - 0.4
    # but don't know the scaling function...
    return ovo

def blockiness(rotated, upper_block_limit=0.9, lower_block_limit=0.1):
    # upper/lower blockiness metrics from Tomato Analyzer, based on a rotated contour
    rowsums = np.sum(rotated, 1)/255
    y1 = np.where(rowsums > 0)[0][0]
    y2 = np.where(rowsums > 0)[0][-1]
    h = y2 - y1
    W_mid = rowsums[int(y1 + h/2)]
    W_upper = rowsums[int(y2 - upper_block_limit*h)]
    W_lower = rowsums[int(y2 - lower_block_limit*h)]
    # proximal is upper/mid, distal is lower/mid
    return W_upper/W_mid, W_lower/W_mid

def minInscribedRect(rotated):
    Amax = 0
    for i in range(0, rotated.shape[0]):
        # rotated masks are not binary, use 128 as cutoff
        ind = np.argwhere(rot[i,:] > 128)
        if not ind.size > 0:
            continue
        # draw an hline, pull L and R endpoints, draw vlines, find their endpoints
        left = ind[0][0]
        right = ind[-1][0]
        hmax_l = np.argwhere(rot[:,left] > 128)[-1][0]
        hmax_r = np.argwhere(rot[:,right] > 128)[-1][0]
        hmax = min(hmax_l, hmax_r)-i
        A = (right - left + 1) * hmax
        if A > Amax:
            Amax = A
            x,y,w,h = (left,i,right-left+1,hmax)
    return x,y,w,h

def validate_output_df(output_df):
    # can do this at runtime, but doing it last is useful for adjusting which columns are shown
    df = output_df.copy()
    if 'area_cm2' in df.columns:
        size_col = 'area_cm2'
    else:
        size_col = 'area_px2'
    if 'row' in df.columns:
        loc_columns = ['row','column']
    else:
        loc_columns = ['center_px_x', 'center_px_y']
    # flag any low solidity berries, should all be >0.97
    if any(df['solidity'] < 0.95):
        print('Berrie(s) with low solidity found:')
        print(df[df['solidity'] < 0.95][['filename'] + loc_columns + ['solidity']])
    # flag excessive z-scores 
    df['mean_size'] = df.groupby('filename')[size_col].transform(lambda x: x.mean())
    df['Z'] = df.groupby('filename')[size_col].transform(lambda x: (x - x.mean()) / x.std())
    if any(abs(df['Z']) > 3):
        print('Object(s) >3 standard deviations from mean found:')
        print(df[abs(df['Z']) > 3][['filename'] + loc_columns + [size_col] + ['mean_size','Z']])

class SingleObject():
    def __init__(self, contr, i):
        self.contr = contr
        self.i = i
        self.obb = orientedBoundingBox(contr)
        self.rotated = rotateContour(contr)
    def basic_shape(self):
        pass

class Transect(SingleObject):
    def __init__(self, contr, i):
        SingleObject.__init__(self, contr, i)
        x,y,w,h = cv2.boundingRect(contr)
        self.w = w
        self.h = h
        self.roundness = roundnessContour(contr)

class Nut(SingleObject):
    def __init__(self, contr, i):
        SingleObject.__init__(self, contr, i)
        if i < 2:
            self.view = 'side'
        else:
            self.view = 'top'
        if 0 < i < 3:
            self.orientation = 'vertical'
        else:
            self.orientation = 'horizontal'

class Kernel(SingleObject):
    def __init__(self, contr, i):
        SingleObject.__init__(self, contr, i)
        self.contr = contr
        self.i = i
        if i < 2:
            self.view = 'dorsal'
        else:
            self.view = 'ventral'
        if 0 < i < 4:
            self.orientation = 'vertical'
        else:
            self.orientation = 'horizontal'
    def analyze_color(self, lab, color_chip_mat):
        L,a,b = np.split(lab, 3, axis=2)
        kernel_mask = np.zeros_like(L)
        cv2.drawContours(kernel_mask, [self.contr], -1, 255, cv2.FILLED)
        # TODO - rewrite this to be applying median across an axis
        # couldn't get it to work correctly
        median_L = np.median(L[np.where(kernel_mask)])
        median_a = np.median(a[np.where(kernel_mask)])
        median_b = np.median(b[np.where(kernel_mask)])
        median_lab = [median_L, median_a, median_b]
        # minimize Euclidean distance to Lab values of Munsell chips
        # define chip values in main() and pass to this function
        self.chip_value = np.argmin(np.apply_along_axis(lambda x: np.sum((x-median_lab)**2), 2, color_chip_mat)) + 1
    def measure_wrinkles(self, gray):
        # take crops around kernel of interest, otherwise running meijering filter takes forever
        kernel_mask = np.zeros_like(gray)
        cv2.drawContours(kernel_mask, [self.contr], -1, 255, cv2.FILLED)
        x,y,w,h = cv2.boundingRect(self.contr)
        crop_gray = gray[y:(y+h),x:(x+w)]
        crop_mask = kernel_mask[y:(y+h),x:(x+w)]
        # apply meijering filter, then crop to remove "halo"
        meij = filters.meijering(cv2.bitwise_and(crop_gray, crop_mask),
                                black_ridges=True,
                                sigmas = range(2,4))
        meij = pcv.apply_mask(meij, crop_mask, mask_color = 'black')
        self.mean_wrinkle = 255 * np.sum(meij) / np.sum(crop_mask)

class VoucherImage():
    def __init__(self, Path):
        self.img, self.path, self.filename = pcv.readimage(str(Path))
        self.gray = pcv.rgb2gray(self.img)
        self.lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        # munsell color-chip matrix for color quantification
        color_chips_rgb = np.array([
            [223,200,150],
            [202,171,128],
            [179,142,110],
            [154,115,89],
            [130,87,69],
            [105,61,52]], dtype=np.uint8)
        color_chips_rgb = np.expand_dims(color_chips_rgb, 1)
        self.color_chips_lab = cv2.cvtColor(color_chips_rgb, cv2.COLOR_RGB2LAB)

    def thresh_foreground(self):
        # threshold the foreground and reuse throughout class
        # saturation on BG is almost all under 10, but "halo" around kernels can be higher
        thr_sat = pcv.threshold.binary(self.hsv[:,:,1], 30, object_type='light')
        # dark spots on the shells have low saturation, grab with value thresh
        thr_val = pcv.threshold.binary(self.hsv[:,:,2], 100, object_type='dark')
        thr_fg = cv2.bitwise_or(thr_sat, thr_val)
        thr_fg = pcv.fill(thr_fg, 5000)
        thr_fg = pcv.fill_holes(thr_fg)
        # try dropping the color card, about a million pixels vs. like 30k for the nuts/kernels
        thr_fg_largeonly = pcv.fill(thr_fg.copy(), 700000)
        thr_fg = thr_fg - thr_fg_largeonly
        self.thr_fg = thr_fg
        self.fg_conts, hierarchy = cv2.findContours(thr_fg, 0, 2)

    def thresh_blue_foreground(self):
        # threshold the foreground and reuse throughout class
        nonblue = np.abs(self.hsv[:,:,0].astype(np.int32) - 103)
        nonblue = nonblue.astype(np.uint8)
        thr_highblue = pcv.threshold.binary(nonblue, 15, object_type='dark')
        thr_sat = pcv.threshold.binary(self.hsv[:,:,1], 70, object_type='light')
        thr_bg = cv2.bitwise_and(thr_highblue, thr_sat)
        thr_fg = pcv.invert(thr_bg)
        thr_fg = pcv.fill(thr_fg, 3000)
        thr_fg = pcv.fill_holes(thr_fg)
        # try dropping the color card, about a million pixels vs. like 30k for the nuts/kernels
        thr_fg_largeonly = pcv.fill(thr_fg.copy(), 700000)
        thr_fg = thr_fg - thr_fg_largeonly
        # clean it up a bit
        thr_fg = cv2.morphologyEx(thr_fg, cv2.MORPH_OPEN, kernel = np.ones((3,3)))
        thr_fg = pcv.fill(thr_fg, 3000)
        self.thr_fg = thr_fg
        self.fg_conts, hierarchy = cv2.findContours(thr_fg, 0, 2)

    def find_text_card(self):
        # use Otsu's, works well for text
        ret, thr_bw = cv2.threshold(self.gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        # clean up specks
        thr_bw = cv2.morphologyEx(thr_bw, cv2.MORPH_OPEN, kernel=np.ones((3,3)))
        # vertical structuring element, since we know the text is vertical
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 600))
        dilation = cv2.dilate(thr_bw, kernel)
        # fill in very large objects
        dilation_nosmall = pcv.fill(dilation, 400000)
        dilation_smallonly = dilation - dilation_nosmall
        text_cont, hierarchy = cv2.findContours(dilation_smallonly, 0, 2)
        text_box_cands = [cv2.boundingRect(x) for x in text_cont]
        text_box = [x for x in text_box_cands if x[1]< 1200 and x[2] < 300 and x[3] > 1000]
        if not text_box:
            print('Issue in image %s\nNo text boxes found!' % self.filename)
            self.card_text = 'NA'
            self.edit_distance_card = 'NA'
        else:
            if len(text_box) == 1:
                text_box = text_box[0]
            else:
                print('Issue in text box location: %s candidate boxes found' % len(text_box))
            x,y,w,h = text_box
            text_crop = self.gray[(y):(y+h),(x):(x+w)]
            text_rotated = cv2.rotate(text_crop, cv2.ROTATE_90_CLOCKWISE)
            # Gaussian blur improves text parsing enormously
            blurred = cv2.GaussianBlur(text_rotated, (11,11), 0)
            self.card_text = pytesseract.image_to_string(blurred).strip()
            # get the edit distance after removing the whitespace
            str1 = re.sub('\s+',' ',self.card_text)
            str2 = re.sub('\s+',' ',self.filename.rstrip('.jpg'))
            self.edit_distance_card = lev(str1, str2)

    def find_size_marker(self):
        def in_top_right(contr, img):
            M = cv2.moments(contr)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            return (cy < img.shape[0]*0.33 and cx > img.shape[1]*0.67)
        # PlantCV's off-the-shelf function is very very rigid, only works with single marker
        potential_markers = [(i,x) for (i,x) in enumerate(self.fg_conts) if roundnessContour(x) > 0.95 and in_top_right(x, self.img)]
        if len(potential_markers) == 0:
            raise Exception('No candidate size markers found')
        elif len(potential_markers) > 1:
            raise Exception('More than one candidate size marker found')
        else:
            i,marker = potential_markers[0]
        x,y,w,h = cv2.boundingRect(marker)
        self.marker_contour = marker
        self.mm_per_px = 24.26 / np.mean((w,h))
        self.marker_i = i
    
    def find_nuts(self):
        # have foreground contours already
        # drop the size marker, run K means, should have top and bottom row
        # only use the size marker if we have one, otherwise just proceed
        if self.marker_i:
            candidate_nuts = [x for i,x in enumerate(self.fg_conts) if i != self.marker_i]
        else:
            candidate_nuts = self.fg_conts
        y_vals = [cv2.moments(x)['m01']/cv2.moments(x)['m00'] for x in candidate_nuts]
        y_vals = np.array(y_vals).reshape(-1,1)
        km = KMeans(init='k-means++', n_clusters=3, n_init=10).fit(y_vals)
        top_row_conts = [x for (i,x) in enumerate(candidate_nuts) if km.labels_[i] == np.argmin(km.cluster_centers_)]
        bottom_row_conts = [x for (i,x) in enumerate(candidate_nuts) if km.labels_[i] == np.argmax(km.cluster_centers_)]
        if len(top_row_conts) != 4:
            raise Warning('Issue in image %s\nShould be 4 objects in top row, %s found' % (self.filename, len(top_row_conts)))
        if len(bottom_row_conts) != 5:
            raise Warning('Issue in image %s\nShould be 5 objects in bottom row, %s found' % (self.filename, len(top_row_conts)))
        # sort them by x-value
        top_row_x = [cv2.moments(x)['m10']/cv2.moments(x)['m00'] for x in top_row_conts]
        bottom_row_x = [cv2.moments(x)['m10']/cv2.moments(x)['m00'] for x in bottom_row_conts]
        top_row_conts = [top_row_conts[i] for i in np.argsort(top_row_x)]
        bottom_row_conts = [bottom_row_conts[i] for i in np.argsort(bottom_row_x)]
        self.nut_contours = top_row_conts + bottom_row_conts
        self.obb_list = [orientedBoundingBox(c) for c in self.nut_contours]
        # create berry objects
        self.nut_list = [Nut(c,i) for (i,c) in enumerate(top_row_conts)]
        self.kernel_list = [Kernel(c,i) for (i,c) in enumerate(bottom_row_conts) if i != 2]
        self.transect = Transect(bottom_row_conts[2],2)
        
    def build_kernel_df(self):
        # defining columns one at a time like this is a bit clunky
        # but it's legible and explicit, which is useful
        kernel_df = pd.DataFrame()
        kernel_df['pos'] = [x.i+1 for x in self.kernel_list]
        kernel_df['view'] = [x.view for x in self.kernel_list]
        kernel_df['orientation'] = [x.orientation for x in self.kernel_list]
        kernel_df['MaxHeight'] = [x.obb[1][1] * self.mm_per_px for x in self.kernel_list]
        kernel_df['MaxWidth'] = [x.obb[1][0] * self.mm_per_px for x in self.kernel_list]
        kernel_df['FSIEI'] = kernel_df['MaxHeight']/kernel_df['MaxWidth']
        kernel_df['Ovoid'] = [ovoid(x.rotated) for x in self.kernel_list]
        kernel_df['MunsellColor'] = [x.chip_value for x in self.kernel_list]
        kernel_df['Wrinkling'] = [x.mean_wrinkle for x in self.kernel_list]
        # # insert file/path last so single value nicely scales up to n rows
        kernel_df.insert(0, 'edit_distance', self.edit_distance_card)
        kernel_df.insert(0, 'card_text', self.card_text)
        kernel_df.insert(0, 'filename', self.filename)
        kernel_df.insert(0, 'path', self.path)
        self.kernel_df = kernel_df

    def build_nut_df(self):
        # defining columns one at a time like this is a bit clunky
        # but it's legible and explicit, which is useful
        nut_df = pd.DataFrame()
        nut_df['pos'] = [x.i+1 for x in self.nut_list]
        nut_df['view'] = [x.view for x in self.nut_list]
        nut_df['orientation'] = [x.orientation for x in self.nut_list]
        nut_df['MaxHeight'] = [x.obb[1][1] * self.mm_per_px for x in self.nut_list]
        nut_df['MaxWidth'] = [x.obb[1][0] * self.mm_per_px for x in self.nut_list]
        nut_df['FSIEI'] = nut_df['MaxHeight']/nut_df['MaxWidth']
        nut_df['Ovoid'] = [ovoid(x.rotated) for x in self.nut_list]
        nut_df['Roundness'] = [roundnessContour(x.contr) for x in self.nut_list]
        nut_df['Blockiness_upper'] = [blockiness(x.rotated)[0] for x in self.nut_list]
        nut_df['Blockiness_lower'] = [blockiness(x.rotated)[1] for x in self.nut_list]
        # # insert file/path last so single value nicely scales up to n rows
        nut_df.insert(0, 'edit_distance', self.edit_distance_card)
        nut_df.insert(0, 'card_text', self.card_text)
        nut_df.insert(0, 'filename', self.filename)
        nut_df.insert(0, 'path', self.path)
        self.nut_df = nut_df
    
    def build_transect_df(self):
        transect_df = pd.DataFrame()
        # ntwd (WIDTH) is along plane of suture
        # ntht (HEIGHT) is orthogonal to suture
        # I find this confusing, but this is the standard
        transect_df['ntwd_mm'] = self.transect.h
        transect_df['ntht_mm'] = self.transect.w
        transect_df['roundness'] = self.transect.roundness
        self.transect_df = transect_df

    def measure_kernel_color(self):
        # separate/calculate color channels
        for krn in self.kernel_list:
            krn.analyze_color(self.lab, self.color_chips_lab)
            krn.measure_wrinkles(self.gray)

    def convert_to_mm(self):
        pass

    def save_annotations(self, annot_dir):
        tmp = self.img.copy()
        cv2.drawContours(tmp, self.nut_contours, -1, (0,255,0), 5)
        cv2.drawContours(tmp, self.marker_contour, -1, (0,0,255), 5)
        for obj in self.nut_list + self.kernel_list:
            obb = obj.obb
            obb_coords = cv2.boxPoints(obb)
            obb_coords = np.int0(obb_coords)
            cv2.drawContours(tmp, [obb_coords], 0, (255,0,255), 5)
        outpath = '%s/%s_annotated%s' % (str(annot_dir), Path(self.filename).stem, Path(self.filename).suffix)
        cv2.imwrite(outpath, tmp)

def analyze_single_path(path, arg_ns):
    # separate out directory analysis into a function for easier passing to tqdm/multiprocessing
    voucher = VoucherImage(path)
    if arg_ns.blue:
        voucher.thresh_blue_foreground()
    else:
        voucher.thresh_foreground()
    if arg_ns.size:
        voucher.find_size_marker()
    voucher.find_text_card()
    voucher.find_nuts()
    voucher.measure_kernel_color()
    voucher.build_kernel_df()
    voucher.build_nut_df()
    voucher.build_transect_df()
    if arg_ns.annotated:
        voucher.save_annotations(arg_ns.annotated)
    return (voucher.kernel_df, voucher.nut_df, voucher.transect_df)

def main():
    args_raw = options()
    args = clean_args(args_raw)
    # actual main function
    if args.input.is_dir():
        async_results = []
        all_kernel_dfs = []
        all_nut_dfs = []
        all_transect_dfs = []
        subfiles = args.input.glob('*')
        subfiles = sorted([x for x in subfiles if valid_img(x)])
        # send results to mp pool with tqdm progress bar
        with mp.Pool(processes=args.nproc) as pool:
            for path in subfiles:
                async_res = pool.apply_async(analyze_single_path, (path, args))                
                async_results.append(async_res)
            for result in tqdm(async_results, total=len(async_results), desc='Processing images...'):
                all_kernel_dfs.append(result.get()[0])
                all_nut_dfs.append(result.get()[1])
                all_transect_dfs.append(result.get()[2])
            final_kernel_df = pd.concat(all_kernel_dfs, ignore_index=True)
            final_nut_df = pd.concat(all_nut_dfs, ignore_index=True)
            final_transect_df = pd.concat(all_transect_dfs, ignore_index=True)
    # if just a single image, skip Pool
    else:
        final_kernel_df = analyze_single_path(args.input, args)[0]
        final_nut_df = analyze_single_path(args.input, args)[1]
        final_transect_df = analyze_single_path(args.input, args)[2]
    # validate_output_df(final_df)
    final_kernel_df.to_csv(re.sub('.csv', '_kernel.csv', str(args.output)), index=False)
    final_nut_df.to_csv(re.sub('.csv', '_nut.csv', str(args.output)), index=False)
    final_transect_df.to_csv(re.sub('.csv', '_transect.csv', str(args.output)), index=False)

if __name__ == '__main__':
    main()