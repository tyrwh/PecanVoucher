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
from skimage import filters, segmentation, io
from skimage.color import rgb2hsv, rgb2gray
from Levenshtein import distance as lev

import os
os.environ["OMP_NUM_THREADS"] = "1"
import joblib
from collections import defaultdict
from skimage import segmentation 

from skimage import measure
from skimage.feature import graycomatrix, graycoprops
from skimage.morphology import binary_erosion, binary_dilation, area_opening, area_closing, disk, remove_small_objects
from skimage.measure import find_contours
from scipy.signal import find_peaks, savgol_filter
from scipy.spatial import distance
from scipy.spatial import KDTree
from skimage.morphology import skeletonize
from skimage import morphology, color, util
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from shapely.affinity import scale
from skimage.draw import polygon
from scipy.ndimage import label


def options():
	parser = argparse.ArgumentParser(description='Pecan Voucher Image Analysis V1.')
	parser.add_argument('-i', '--input', help='Input file(s). Must be valid image or directory with valid images.', required=True)
	parser.add_argument('-g', '--groove_model', help='groove model file. Must be valid .sav model file.', required=True)
	parser.add_argument('-st', '--shell_thickness_model', help='shell thickness model file. Must be valid .sav model file.', required=True)
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

def valid_model(Path):
	return Path.suffix in ['.sav']

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
	
	# Validate models
	if not args.shell_thickness_model or not valid_model(Path(args.shell_thickness_model)):
		raise Exception(f"Invalid shell thickness model: {args.shell_thickness_model}")
	if not args.groove_model or not valid_model(Path(args.groove_model)):
		raise Exception(f"Invalid groove model: {args.groove_model}")
	
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
		self.x = x
		self.y = y
		self.w = w
		self.h = h
		self.roundness = roundnessContour(contr)
		self.tile = None
		self.kernel_mask = None
		self.groove_mask = None
		self.groove_metrics = None
		self.shell_mask = None
		self.shell_width = None
		self.mosaic_img = None

	def create_mosaic(self):
		
		def ensure_3d(mask):
			if mask.ndim == 2:
				mask = np.expand_dims(mask, axis=-1)  # Add a channel dimension (shape [height, width, 1])
				mask = np.repeat(mask, 3, axis=-1)  # Repeat the mask along the color channels (shape [height, width, 3])
				mask = mask.astype(np.uint8)  # Ensure the mask is of type uint8 (binary values 0 or 1)
			return mask

		# Apply this function to all your masks
		kernel_mask = ensure_3d(self.kernel_mask) * 255
		groove_mask = ensure_3d(self.groove_mask) * 255
		shell_mask = ensure_3d(self.shell_mask) * 255

		tile_rgb = cv2.cvtColor(self.tile, cv2.COLOR_BGR2RGB)
		top_row = np.hstack((tile_rgb, kernel_mask))
		bottom_row = np.hstack((groove_mask, shell_mask))
		self.mosaic_img = np.vstack((top_row, bottom_row))

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
	def __init__(self, Path, g_model_path, st_model_path):
		self.img, self.path, self.filename = pcv.readimage(str(Path))
		self.gray = pcv.rgb2gray(self.img)
		self.lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
		self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
		self.kernel_model = g_model_path #'C:/Users/des346/Desktop/Code Projects/Pecan/pecan_model.sav'
		self.shell_model = st_model_path #'C:/Users/des346/Desktop/Code Projects/Pecan/pecan_shell_model_2.sav'
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

	def analyze_transect(self):
		
		self.transect.tile = crop_image_centered_on_contr(self.img, self.transect)
		self.transect.kernel_mask, kernel_contour = segment_kernels(self.transect.tile, self.kernel_model, self.transect.contr)
		self.transect.groove_mask, self.transect.groove_metrics = segment_grooves(kernel_contour)
		self.transect.shell_mask, self.transect.shell_width = segment_shell(self.transect.tile, self.shell_model, self.transect.kernel_mask, self.transect.contr)
		self.transect.create_mosaic()
		plt.imshow(self.transect.mosaic_img)
		plt.show()
		
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
		# ntwd (WIDTH) is along plane of suture
		# ntht (HEIGHT) is orthogonal to suture
		# I find this confusing, but this is the standard
		data = {
			'ntwd_mm': self.transect.h,
			'ntht_mm': self.transect.w,
			'roundness': self.transect.roundness,
			'shell width': self.transect.shell_width * self.mm_per_px
		}
		# Iterate over the groove metrics and correctly format the column names
		for index, groove in enumerate(self.transect.groove_metrics, start=1):  # Start at 1 for human-friendly labels
			for metric in groove:
				metric_name = metric[0]  # Extract metric name (e.g., "Groove Width")
				value = metric[1] # Extract value
				# Assign to DataFrame with proper formatting
				data[f"{metric_name} {index}"] = value

		transect_df = pd.DataFrame([data])
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
	voucher = VoucherImage(path, arg_ns.groove_model, arg_ns.shell_thickness_model)
	if arg_ns.blue:
		voucher.thresh_blue_foreground()
	else:
		voucher.thresh_foreground()
	if arg_ns.size:
		voucher.find_size_marker()
	voucher.find_text_card()
	voucher.find_nuts()
	voucher.analyze_transect()
	voucher.measure_kernel_color()
	voucher.build_kernel_df()
	voucher.build_nut_df()
	voucher.build_transect_df()
	if arg_ns.annotated:
		voucher.save_annotations(arg_ns.annotated)
	return (voucher.kernel_df, voucher.nut_df, voucher.transect_df)

class TransectTile():
	def __init__(self, image, model_path, transect_contour):
		"""
		Initialize the TransectTile object with an image path.
		Loads the image and stores it as an instance attribute.
		"""
		self.image = image
		if self.image is None:
			raise ValueError(f"Error loading image: {image}")
		
		self.model_path = model_path
		self.model = joblib.load(self.model_path)
		if self.model is None:
			raise ValueError(f"Error loading model: {model_path}")
		
		self.transect_contour = transect_contour
		if self.transect_contour is None:
			raise ValueError(f"Error loading transect contour")
		
		self.tile_height = 500
		self.tile_width = 500
		self.contour_mask = None
		self.segments = None  # To store SLIC segmentation result
		self.features = None  # To store calculated features
		self.model_features = None
		self.predictions = None
		self.raw_contours = None
		self.final_mask = None
		self.groove_metrics = []
		self.groove_mask = None
		self.shell_width = None
	
	def create_slic(self, n_segment, compactness, start_label, contour):

		if (contour is None):
			self.segments = segmentation.slic(self.image, n_segments=n_segment, convert2lab=True, enforce_connectivity=True, spacing=None, compactness=compactness, start_label=start_label)
		else:
			# Generate SLIC superpixels
			self.segments = segmentation.slic(self.image, n_segments=n_segment, convert2lab=True, enforce_connectivity=True, spacing=None, compactness=compactness, start_label=start_label, mask=contour)

	
	def calc_texture_features(self, intensity_image):
	
		if len(intensity_image.shape) == 3:  # Ensure the image is grayscale
			raise ValueError("Texture features require a grayscale image.")
		
		# Calculate co matrix
		glcm = graycomatrix(intensity_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

		# Calculate specific Haralick features using greycoprops
		contrast = graycoprops(glcm, 'contrast')
		dissimilarity = graycoprops(glcm, 'dissimilarity')
		homogeneity = graycoprops(glcm, 'homogeneity')
		energy = graycoprops(glcm, 'energy')
		correlation = graycoprops(glcm, 'correlation')
		ASM = graycoprops(glcm, 'ASM')

		# Create list of key value pairs
		texture_features = [float(contrast.item()),
			float(dissimilarity.item()),
			float(homogeneity.item()),
			float(energy.item()),
			float(correlation.item()),
			float(ASM.item())
		]

		return texture_features
	
	def calc_features(self):
		self.features = [['ID','intensity_max: Red','intensity_min: Red','intensity_mean: Red','intensity_std: Red','centroid: X','centroid: Y','intensity_max: Green','intensity_min: Green',
					'intensity_mean: Green','intensity_std: Green','centroid: X','centroid: Y','intensity_max: Blue','intensity_min: Blue','intensity_mean: Blue','intensity_std: Blue',
					'centroid: X','centroid: Y','intensity_max: Hue','intensity_min: Hue','intensity_mean: Hue','intensity_std: Hue','centroid: X','centroid: Y','intensity_max: Saturation',
					'intensity_min: Saturation','intensity_mean: Saturation','intensity_std: Saturation','centroid: X','centroid: Y','intensity_max: Value','intensity_min: Value',
					'intensity_mean: Value','intensity_std: Value','centroid: X','centroid: Y','contrast','dissimilarity','homogeneity','energy','correlation','ASM']]

		grayscale = rgb2gray(self.image)
		grayscale = (grayscale * 255).astype(np.uint8)
		red_channel = self.image[...,0]
		green_channel = self.image[...,1]
		blue_channel = self.image[...,2]

		# Convert RGB to HSV
		image_hsv = rgb2hsv(self.image)

		# Extract individual HSV channels
		hue_channel = image_hsv[:, :, 0]       # Hue channel
		saturation_channel = image_hsv[:, :, 1]  # Saturation channel
		value_channel = image_hsv[:, :, 2]       # Value (brightness) channel
		
		image_channels = [['Red', red_channel], ['Green', green_channel], ['Blue', blue_channel], ['Hue', hue_channel], ['Saturation', saturation_channel], ['Value', value_channel]]

		grayscale_regions = measure.regionprops(self.segments, intensity_image=grayscale)

		channel_regions = []
		for channel_data in image_channels:
			regions = measure.regionprops_table(self.segments, intensity_image=channel_data[1], properties=('intensity_max', 'intensity_min', 'intensity_mean', 'intensity_std', 'centroid'))
			channel_regions.append(regions)

		for num_region, region in enumerate(grayscale_regions):
			intensity_image = grayscale_regions[num_region]['intensity_image']
			texture_features = self.calc_texture_features(intensity_image)
			segment_features = [num_region]

			for channel_region in channel_regions:
				
				channel_features = [float(channel_region['intensity_max'][num_region]),
									float(channel_region['intensity_min'][num_region]),
									float(channel_region['intensity_mean'][num_region]),
									float(channel_region['intensity_std'][num_region]),
									float(channel_region['centroid-1'][num_region]),
									float(channel_region['centroid-0'][num_region]),#]
				]
				
				for channel_feature in channel_features:
					segment_features.append(channel_feature)
				
			
			for texture_feature in texture_features:
				segment_features.append(texture_feature)
			
			
			self.features.append(segment_features)
	
	def clean_features(self):
		"""
		Converts a list of lists into a pandas DataFrame using the first row as column names, 
		drops specified features (columns), and returns the DataFrame.

		Args:
			data (list of lists): Original data where the first sublist contains column names.

		Returns:
			pd.DataFrame: The DataFrame after dropping specified features.
		"""

		dropped_features = ['ID','centroid: X', 'centroid: Y', 'intensity_max: Hue', 'intensity_mean: Hue', 'intensity_min: Hue', 'intensity_std: Hue']

		# Extract column names from the first row of the list
		column_names = self.features[0]
		
		# Extract the remaining rows as data
		rows = self.features[1:]
		
		# Convert to a DataFrame
		df = pd.DataFrame(rows, columns=column_names)
		
		# Drop the specified features
		self.model_features = df.drop(columns=dropped_features, errors='ignore')
	
	def run_prediction_model(self):
		self.predictions = self.model.predict(self.model_features)
	
	def adjust_predictions(self):
		features_list = self.model_features[1:]
		for index, feature in features_list.iterrows():
			if feature.iloc[14] < 0.1:
				self.predictions[index] = 2
	
	def create_mask(self):
		"""
		Creates an image where each SLIC region is replaced with its predicted class.

		Args:
			predictions (list): List of predicted class labels for each SLIC.
			slics (2D array): Matrix where each value corresponds to a SLIC label.

		Returns:
			2D array: A matrix where each pixel is replaced by its predicted class label.
		"""
		# Check if predictions match the unique SLIC labels
		unique_slic_labels = np.unique(self.segments)
		unique_slic_labels = unique_slic_labels[unique_slic_labels != 0]
		if len(self.predictions) != len(unique_slic_labels):
			raise ValueError("Number of predictions does not match the number of unique SLIC labels.")
		
		# Map SLIC labels to predicted classes
		#slic_to_class = {label: self.predictions[label] for label in unique_slic_labels}
		slic_to_class = {label: self.predictions[i] for i, label in enumerate(unique_slic_labels)}

		# Create a classified image by replacing each SLIC label with its class label
		classified_image = np.vectorize(slic_to_class.get)(self.segments)

		# Create a binary mask: 255 for class 1, 0 for all other classes
		binary_mask = np.where(classified_image == 1, 255, 0).astype(np.uint8)
		'''
		fig = plt.figure(frameon=False)
		fig.set_size_inches(binary_mask.shape[1] / 100, binary_mask.shape[0] / 100)  # Set size matching the image dimensions

		# Add the axes without any frame
		ax = plt.Axes(fig, [0, 0, 1, 1])  # Position the image to occupy the full figure
		ax.set_axis_off()  # Remove axis
		fig.add_axes(ax)
		'''
		return binary_mask
	
	def process_mask(self, mask, erosion_size=3, area_threshold=10000):
	
		# Create a disk-shaped structuring element for erosion and dilation
		structuring_element = disk(erosion_size)
		
		# Perform erosion
		eroded_mask = binary_erosion(mask, structuring_element)
		
		# Perform area opening to remove small objects
		opened_mask = area_opening(eroded_mask, area_threshold=area_threshold)

		# Perform dilation to counter the effect of erosion
		dilated_mask = binary_dilation(opened_mask, structuring_element)

		# Perform area closing to remove small holes
		self.final_mask = area_closing(dilated_mask, area_threshold=area_threshold)

		def split_binary_mask():
			# Create copies of the mask
			top_half_mask = self.final_mask.copy()
			bottom_half_mask = self.final_mask.copy()

			# Make the bottom half black in the first mask
			bottom_half_mask[self.tile_height // 2 :, :] = 0

			# Make the top half black in the second mask
			top_half_mask[: self.tile_height // 2, :] = 0

			return bottom_half_mask, top_half_mask
		
		self.top_mask, self.btm_mask = split_binary_mask()
		
		# Find contours
		self.raw_contours = find_contours(self.top_mask, level=0.5)
		self.raw_contours += find_contours(self.btm_mask, level=0.5)

	def process_shell_mask(self, shell_mask, kernel_mask, erosion_size=3, area_threshold=3000):

		mask1 = shell_mask.astype(np.uint8) * 255
		mask2 = kernel_mask.astype(np.uint8) * 255
		subtracted_mask = cv2.subtract(mask1, mask2)
	
		# Create a disk-shaped structuring element for erosion and dilation
		structuring_element = disk(erosion_size)
		
		# Perform erosion
		eroded_mask = binary_erosion(subtracted_mask, structuring_element)
		
		# Perform area opening to remove small objects
		opened_mask = area_opening(eroded_mask, area_threshold=area_threshold)

		# Perform dilation to counter the effect of erosion
		dilated_mask = binary_dilation(opened_mask, structuring_element)

		# Perform area closing to remove small holes
		self.final_mask = area_closing(dilated_mask, area_threshold=area_threshold)

	def find_average_shell_width(self, smallest_len=7, largest_len=27, num_samples=100):
		"""
		Finds the average width of a pecan shell by calculating the distance from the outer contour
		to the nearest point of the inner contour using a sampled subset of points.

		Parameters:
		- num_samples: Number of points to sample from the outer contour.
		- smallest_len: Distance in pixels that the smallest connected points from the outer contour to inner contour must be to be considered valid
		- largest_len: Distance in pixels that the largest connected points from the outer contour to inner contour must be to be considered valid

		Returns:
		- average_width: The average width of the shell.
		"""
		final_mask = self.final_mask.astype(np.uint8) * 255

		# Find contours using RETR_TREE to capture both inner and outer contours
		contours, __ = cv2.findContours(final_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		largest_contour = max(contours, key=cv2.contourArea)

		mask = np.zeros(final_mask.shape, dtype=np.uint8)
		cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

		if cv2.contourArea(largest_contour) < 50000:

			def get_skeleton_endpoints(skeleton):
				"""Finds endpoints of a skeletonized shape."""
				y_indices, x_indices = np.where(skeleton == 255)
				endpoints = []

				# Check each pixel to see if it's an endpoint
				for x, y in zip(x_indices, y_indices):
					# Extract a 3x3 neighborhood
					neighbors = skeleton[y - 1:y + 2, x - 1:x + 2]
					
					# Count nonzero pixels (excluding the center pixel)
					num_neighbors = np.count_nonzero(neighbors) - 1
					
					if num_neighbors == 1:  # Endpoint has exactly one neighbor
						endpoints.append((x, y))

				return endpoints
			
			skeleton = skeletonize(mask // 255, method='lee').astype(np.uint8) * 255
			
			pruned_skeleton, __, ___ = pcv.morphology.prune(skel_img=skeleton, size=150)

			endpoints = get_skeleton_endpoints(pruned_skeleton)

			def draw_line_between_endpoints(mask, endpoints):
				"""Draws a 3-pixel-thick line between the two farthest endpoints."""
				if len(endpoints) < 2:
					return mask  # Not enough endpoints to connect

				cv2.line(mask, endpoints[0], endpoints[1], 255, thickness=1)

				return mask
			
			edited_mask = draw_line_between_endpoints(mask, endpoints)

		else:
			edited_mask = final_mask

		contours_edited, ___ = cv2.findContours(edited_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		if len(contours_edited) < 2:
			raise ValueError("The image must contain both inner and outer contours.")

		# Sort contours by area to identify the outer and inner contours
		contours_edited = sorted(contours_edited, key=cv2.contourArea, reverse=True)
		outer_contour = contours_edited[0].squeeze()  # First contour is the outer
		inner_contour = contours_edited[1].squeeze()  # Second contour is the inner

		# Uniformly sample points from the outer contour
		if len(outer_contour) > num_samples:
			step = max(1, len(outer_contour) // num_samples)
			sampled_outer_contour = outer_contour[::step][:num_samples]
		else:
			sampled_outer_contour = outer_contour

		# Build KDTree for the inner contour
		inner_tree = KDTree(inner_contour)

		# Calculate distances and find corresponding points
		distances = []
		connections = []  # Store pairs of outer and nearest inner points
		for point in sampled_outer_contour:
			distance, index = inner_tree.query(point)
			if smallest_len < distance < largest_len:
				distances.append(distance)
				connections.append((point, inner_contour[index]))

		# Compute the average width of the donut
		average_width = np.mean(distances)
		if len(distances) < 20:
			average_width = "null"

		'''
		plt.figure(figsize=(8, 8))
		plt.imshow(self.final_mask, cmap='gray')
		plt.axis('off')
		plt.title("Outer to Inner Contour Connections")
		# Plot contours
		plt.plot(outer_contour[:, 0], outer_contour[:, 1], 'b-', label='Outer Contour')
		plt.plot(inner_contour[:, 0], inner_contour[:, 1], 'r-', label='Inner Contour')
		# Plot connections
		for outer_point, inner_point in connections:
			plt.plot([outer_point[0], inner_point[0]], [outer_point[1], inner_point[1]], 'g--', lw=0.7)
		plt.legend()
		plt.show()
		'''
		self.shell_width = average_width

	def process_annotation(self, polygon_coordinates):
	
		length = len(polygon_coordinates)
		polygon_coordinates = np.flip(polygon_coordinates)
		flipped = False
		
		# Check random points to see which side of image the annotation is on
		a = polygon_coordinates[round(length/3)]
		b = polygon_coordinates[round(length/3*2)]
		c = polygon_coordinates[length-10]
		avg_Y = (a[1] + b[1] + c[1]) / 3

		# Invert Y values for bottom kernels
		corrected_coordinates = []
		corrected_Y_values = []
		
		if avg_Y < self.tile_height / 2:
			for entry in polygon_coordinates:
				y_value = self.tile_height - entry[1]
				corrected_coordinates.append([entry[0], y_value])
				corrected_Y_values.append(y_value)
				flipped = True
		else:
			for entry in polygon_coordinates:
				y_value = entry[1]
				corrected_coordinates.append([entry[0], y_value])
				corrected_Y_values.append(y_value)
		
		# Find where to truncate coordinates 
		lowest_pt_X = [self.tile_height, 0]
		furthest_pt_X = [0, 0]
		starting_pt = 0
		ending_pt = 0
		
		for index, point in enumerate(corrected_coordinates):
			# Find lowest X value for starting point
			if point[0] < lowest_pt_X[0] or (point[0] == lowest_pt_X[0] and point[1] > lowest_pt_X[1]):
				lowest_pt_X = [point[0],point[1]]
				starting_pt = index
			# Find highest X value for ending point
			elif point[0] > furthest_pt_X[0] or (point[0] == furthest_pt_X[0] and point[1] > furthest_pt_X[1]):
				furthest_pt_X = [point[0],point[1]]
				ending_pt = index

		# Truncate coordinates
		truncated_coordinates = []
		truncated_Y_coords = []
		
		if ending_pt < starting_pt:
			truncated_coordinates = corrected_coordinates[ending_pt + 1:starting_pt-1]
			truncated_Y_coords = corrected_Y_values[ending_pt + 1:starting_pt-1]
			if flipped == False:
				truncated_coordinates = corrected_coordinates[starting_pt:length-1] + corrected_coordinates[0:ending_pt]
				truncated_Y_coords = corrected_Y_values[starting_pt:length-1] + corrected_Y_values[0:ending_pt]
		else:
			truncated_coordinates = corrected_coordinates[starting_pt:ending_pt]
			truncated_Y_coords = corrected_Y_values[starting_pt:ending_pt]
		
		return truncated_coordinates, truncated_Y_coords
	
	def create_grooves(self, coordinates, y_coords):
		# Convert list to numpy array
		x = np.array(y_coords)
		
		def find_three_peaks(x, height_step=10, distance_step=5, prominence_step=1):
			"""
			Adjust height, distance, and prominence iteratively to find exactly 3 peaks in the data.

			Parameters:
				x (array-like): The input data to find peaks in.
				initial_height (float): Initial height threshold for peak detection.
				initial_distance (float): Initial distance threshold for peak detection.
				initial_prominence (float): Initial prominence threshold for peak detection.
				height_step (float): Step size for adjusting height.
				distance_step1 (float): Step size for the first distance adjustment.
				distance_step2 (float): Step size for the second distance adjustment.
				distance_step3 (float): Step size for the third distance adjustment.
				prominence_step (float): Step size for adjusting prominence.

			Returns:
				tuple: A tuple containing the peaks and their properties.
			"""
			
			'''
			Parameters Info
			0th index = height
			1st index = prominence
			2nd index = distance
			'''

			parameters = [370, 10, 20]

			peaks, properties = find_peaks(x, height=parameters[0], prominence=parameters[1], distance=parameters[2])

			for i in range(10):

				if len(peaks) == 3:
					return peaks

				if len(peaks) > 3:
					# Too many peaks: tighten criteria
					for index, step in enumerate([height_step, prominence_step, distance_step]):
						parameters[index] = max(1, parameters[index] + step)
						peaks, properties = find_peaks(x, height=parameters[0], prominence=parameters[1], distance=parameters[2])
						if len(peaks) <= 3:
							break
				else:
					# Too few peaks: loosen criteria
					for index, step in enumerate([height_step, prominence_step, distance_step]):
						parameters[index] = max(1, parameters[index] - step)
						peaks, properties = find_peaks(x, height=parameters[0], prominence=parameters[1], distance=parameters[2])
						if len(peaks) >= 3:
							break

			return peaks

		peaks = find_three_peaks(x)
		
		# Segment grooves from coordinates
		grooves = []
		polygons = []
		for index, peak in enumerate(peaks[:-1]):
			groove_coordinates = coordinates[peaks[index]:peaks[index+1]]
			grooves.append(groove_coordinates)
			polygon = Polygon(groove_coordinates)
			polygons.append(polygon)
		
		return grooves, polygons
	
	def get_measurements(self, polygon, polygon_coordinates):
	
		last_pt = len(polygon_coordinates) - 1

		# Get groove width
		pt1 = np.array(polygon_coordinates[0])
		pt2 = np.array(polygon_coordinates[last_pt])
		groove_width = np.linalg.norm(pt2 - pt1)

		# Get groove angle
		groove_diff_X = pt2[0] - pt1[0]
		groove_diff_Y = pt2[1] - pt1[1]
		groove_angle_radians = np.arctan2(groove_diff_X, groove_diff_Y)
		groove_angle_degrees = np.degrees(groove_angle_radians)

		# Get groove depth
		# Get center point from groove width line
		ctr_pt_X = pt1[0] + ((pt2[0]-pt1[0])/2)
		ctr_pt_Y = pt1[1] + ((pt2[1]-pt1[1])/2)
		ctr_pt = [ctr_pt_X, ctr_pt_Y]

		# Get lowest Y value point in groove
		lowest_pt = [self.tile_height, self.tile_height]
		for point in polygon_coordinates:
			if point[1] < lowest_pt[1]:
				lowest_pt = point
		ctr_pt = np.array(ctr_pt)
		lowest_pt = np.array(lowest_pt)
		groove_depth = np.linalg.norm(ctr_pt - lowest_pt)

		# Get groove depth angle
		depth_diff_X = ctr_pt[0] - lowest_pt[0]
		depth_diff_Y = ctr_pt[1] - lowest_pt[1]
		depth_angle_radians = np.arctan2(depth_diff_X, depth_diff_Y)
		depth_angle_degrees = np.degrees(depth_angle_radians)

		# Get net area
		net_area = polygon.area

		# Get convex area
		convex_area = polygon.convex_hull.area

		# Calculate solidarity
		solidity = net_area / convex_area

		# Aggregate measurements
		self.groove_metrics.append([
			["Groove Width", float(groove_width)],
			["Groove Width Angle", float(groove_angle_degrees)],
			["Groove Depth", float(groove_depth)],
			["Groove Depth Angle", float(depth_angle_degrees)],
			["Net Area", float(net_area)],
			["Convex Area", float(convex_area)],
			["Solidity", float(solidity)]
		])
	
	def create_groove_mask(self, polygons):
		"""
		Creates a labeled mask from a list of lists of Shapely Polygon objects and plots each groove in a different color.
		
		Parameters:
			tile_height (int): Height of the mask image.
			tile_width (int): Width of the mask image.
			polygons (list): List of lists containing Shapely Polygon objects.
			flip (bool): If True, flips the polygons along the horizontal axis.
			
		Returns:
			ndarray: Labeled mask with each polygon assigned a unique label.
		"""
		# Create an empty labeled mask
		labeled_mask = np.zeros((self.tile_height, self.tile_width), dtype=np.int32)
		
		# Unique label for each polygon
		label = 1
		
		# Process each list of polygons
		for index, polygon_list in enumerate(polygons):
			for poly in polygon_list:
				if index == 0:
					# Flip the polygon along the horizontal axis
					poly = scale(poly, xfact=1, yfact=-1, origin=(0, self.tile_height / 2))
				
				# Get the coordinates of the polygon
				x, y = poly.exterior.coords.xy
				
				# Convert coordinates to integer indices
				rr, cc = polygon(np.array(y, dtype=int), np.array(x, dtype=int), labeled_mask.shape)
				
				# Assign a unique label to the polygon
				labeled_mask[rr, cc] = label
				label += 1

		# Generate a colormap for visualization
		#cmap = ListedColormap(plt.colormaps["tab20"].colors[:label])  # Use a qualitative colormap

		fig = plt.figure(frameon=False)
		fig.set_size_inches(labeled_mask.shape[1] / 100, labeled_mask.shape[0] / 100)  # Set size matching the image dimensions

		# Add the axes without any frame
		ax = plt.Axes(fig, [0, 0, 1, 1])  # Position the image to occupy the full figure
		ax.set_axis_off()  # Remove axis
		fig.add_axes(ax)

		self.groove_mask = labeled_mask

	def erode_shell(self):
		
		gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
		mask = np.zeros_like(gray)
		cv2.drawContours(mask, [self.transect_contour], -1, 255, thickness=-1)

		# Erode the mask by 60 pixels
		eroded_mask = cv2.erode(mask, kernel=np.ones((60, 60), np.uint8), iterations=1)
		
		# Subtract the eroded mask from the original mask
		boundary_mask = cv2.subtract(mask, eroded_mask)

		# Apply the boundary mask to the image
		boundary_image = cv2.bitwise_and(self.image, self.image, mask=boundary_mask)

		return boundary_image, boundary_mask

	def contour_to_mask(self):
		'''
		Converts an OpenCV contour to a binary mask.
		'''

		mask = np.zeros(self.image.shape[:2], dtype=np.uint8)  # Create a blank mask
		cv2.drawContours(mask, [self.transect_contour.contr], -1, color=1, thickness=cv2.FILLED)  # Draw the contour on the mask
		self.contour_mask = mask
	
	def normalize_image(self, image, mask):
		"""
		Applies histogram equalization to each channel of the image only in regions defined by the mask.

		Parameters:
			image (numpy.ndarray): The original RGB image.
			mask (numpy.ndarray): A binary mask where white pixels define the regions of interest.

		Returns:
			equalized_image (numpy.ndarray): The resulting image with histogram equalization applied to masked areas.
		"""
		# Ensure the mask is binary
		mask = (mask > 0).astype(np.uint8)

		# Create a copy of the original image to hold the result
		equalized_image = image.copy()

		# Process each channel independently
		for channel in range(self.image.shape[2]):
			# Extract the channel
			channel_data = image[:, :, channel]

			# Apply mask to extract the relevant region
			masked_region = cv2.bitwise_and(channel_data, channel_data, mask=mask)

			# Equalize the histogram of the masked region
			equalized_region = cv2.equalizeHist(masked_region)

			# Combine the equalized region back into the channel
			equalized_channel = np.where(mask == 1, equalized_region, channel_data)

			# Update the result image
			equalized_image[:, :, channel] = equalized_channel

		self.image = equalized_image

	
def segment_kernels(image, model_path, transect_contour):
	transect = TransectTile(image, model_path, transect_contour)
	mask = np.zeros(transect.image.shape, dtype=np.uint8)
	cv2.fillPoly(mask, [transect_contour], 255)
	mask= cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
	transect.create_slic(n_segment=1000, compactness=10, start_label=1, contour=mask)
	transect.calc_features()
	transect.clean_features()
	transect.run_prediction_model()
	transect.adjust_predictions()
	binary_mask = transect.create_mask()
	transect.process_mask(binary_mask)
	return transect.final_mask, transect

def segment_grooves(transect: TransectTile):
	detected_grooves = []
	for coordinate in transect.raw_contours:
		points, y_coords = transect.process_annotation(coordinate)
		grooves, polygons = transect.create_grooves(points, y_coords)
		detected_grooves.append(polygons)
		for index, groove in enumerate(grooves):
			transect.get_measurements(polygons[index], groove)
	transect.create_groove_mask(detected_grooves)

	return transect.groove_mask, transect.groove_metrics

def segment_shell(image, model_path, kernel_mask, transect_contour):
	transect = TransectTile(image, model_path, transect_contour)
	eroded_shell, binary_shell_mask= transect.erode_shell()
	transect.normalize_image(eroded_shell, binary_shell_mask)
	#image_rgb = cv2.cvtColor(transect.image, cv2.COLOR_BGR2RGB)
	transect.create_slic(n_segment=400, compactness=10, start_label=1, contour=binary_shell_mask)
	transect.calc_features()
	transect.clean_features()
	transect.run_prediction_model()
	shell_mask = transect.create_mask()
	transect.process_shell_mask(shell_mask, kernel_mask)
	transect.find_average_shell_width()
	return transect.final_mask, transect.shell_width

def crop_image_centered_on_contr(uncropped, transect: Transect, crop_size=500):
	"""
	Crops a section of the image centered on the transect contour.

	Args:
		image (ndarray): Original image.
		bbox (tuple): Bounding box (x, y, width, height).
		crop_size (int): Desired size of the cropped square image (default: 500).

	Returns:
		ndarray: Cropped and resized image.
	"""
	image = uncropped
	# Calculate the center of the bounding box
	cx = transect.x + transect.w // 2
	cy = transect.y + transect.h // 2

	def adjust_contour_coordinates(contour, x_subtract, y_subtract):
		'''
		Adjust the x and y coordinates of each point in an OpenCV contour.
		'''

		# Subtract the values from x and y coordinates
		floating_contour = contour - np.array([[[x_subtract, y_subtract]]])
		transect.contr = floating_contour.astype(np.int32)

	shift_right = cx-(crop_size/2)
	shift_left  = cy-(crop_size/2)
	adjust_contour_coordinates(transect.contr, shift_right, shift_left)
	
	# Define the cropping box
	half_size = crop_size // 2
	start_x = max(cx - half_size, 0)
	start_y = max(cy - half_size, 0)
	end_x = min(cx + half_size, image.shape[1])
	end_y = min(cy + half_size, image.shape[0])

	# Adjust crop size to ensure exact dimensions
	if end_x - start_x < crop_size:
		if start_x == 0:
			end_x = min(start_x + crop_size, image.shape[1])
		elif end_x == image.shape[1]:
			start_x = max(0, end_x - crop_size)
	if end_y - start_y < crop_size:
		if start_y == 0:
			end_y = min(start_y + crop_size, image.shape[0])
		elif end_y == image.shape[0]:
			start_y = max(0, end_y - crop_size)

	# Crop the region
	cropped_image = image[start_y:end_y, start_x:end_x]

	# Resize to ensure exact crop size
	cropped_image = cv2.resize(cropped_image, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

	return cropped_image

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