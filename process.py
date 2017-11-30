import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from skimage.feature import hog
import time
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.measurements import label
from IPython import display

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    features = cv2.resize(img, size).ravel()
    return features

# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec, block_norm="L2-Hys")
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec, block_norm="L2-Hys")
        return features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(filenames, config):
    # unpack configs
    color_space = config['color_space']
    spatial_size = config['spatial_size']
    hist_bins = config['hist_bins']
    orient = config['orient']
    pix_per_cell = config['pix_per_cell']
    cell_per_block = config['cell_per_block']
    hog_channel = config['hog_channel']
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in filenames:
        # Read in each one by one
        image = cv2.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            elif color_space == 'Yrb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        else: feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins)

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)

        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features, hog_features)))
    # Return list of feature vectors
    print('image format = 0 - ' + str(np.max(image)))
    print('featureimage format = 0 - ' + str(np.max(feature_image)))
    print('spatial_features.shape = ' + str(spatial_features.shape) )
    print('hist_features.shape = ' + str(hist_features.shape))
    print('hog_features.shape = ' + str(hog_features.shape))
    return features

def find_cars_multi_scales(img, coords, svc, X_scaler, config):

    # check image format
    # print('image format = 0 - ' + str(np.max(img)) )
    draw_img = np.copy(img)
    box_list = []
    energy_list = []
    subimg_list = []
    # unpack configs
    color_space = config['color_space']
    spatial_size = config['spatial_size']
    hist_bins = config['hist_bins']
    orient = config['orient']
    pix_per_cell = config['pix_per_cell']
    cell_per_block = config['cell_per_block']
    hog_channel = config['hog_channel']

    #channel conversion
    if color_space == 'HSV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif color_space == 'HLS':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif color_space == 'Yrb':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    elif color_space == 'YUV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    else: feature_image = np.copy(img)

    # img = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    for i in range(len(coords)):
        # unpack coord
        scale, xstart, xstop, ystart, ystop = coords[i]
        # img = img.astype(np.float32)/255
        # img_tosearch = feature_image[ystart:ystop,xstart:xstop,:]
        # ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        # ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
        ctrans_tosearch = feature_image[ystart:ystop,xstart:xstop,:]
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        # ch1 = ctrans_tosearch[:,:,hog_channel]
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
        nfeat_per_block = orient*cell_per_block**2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window ) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window ) // cells_per_step + 1

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                # hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                # hog_features = hog_feat1
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))


                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                test_prediction = svc.predict(test_features)
                test_score = svc.decision_function(test_features)[0]

                # Drawing
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)

                # draw_img_copy = np.copy(draw_img)
                # if test_prediction == 1:
                #     cv2.rectangle(draw_img,(xbox_left+xstart, ytop_draw+ystart),(xbox_left+win_draw+xstart,ytop_draw+win_draw+ystart),(0,255,0),2)
                #     box_list.append([[xbox_left+xstart,xbox_left+win_draw+xstart],[ytop_draw+ystart,ytop_draw+win_draw+ystart]])
                #     cv2.putText(draw_img, text='{0:.2g}'.format(svc.decision_function(test_features)[0]), org=(xbox_left+xstart,ytop_draw+ystart-10),fontFace=2, fontScale=1, color=(0,255,0), thickness=2)
                #     subimg_list.append(subimg)
                if test_score > 0.0:

                    box_list.append([[xbox_left+xstart,xbox_left+win_draw+xstart],[ytop_draw+ystart,ytop_draw+win_draw+ystart]])
                    energy_list.append(test_score)

                    # cv2.rectangle(draw_img,(xbox_left+xstart, ytop_draw+ystart),(xbox_left+win_draw+xstart,ytop_draw+win_draw+ystart),(0,255,0),3)
                    # cv2.putText(draw_img, text='{0:.2g}'.format(svc.decision_function(test_features)[0]), org=(xbox_left+xstart,ytop_draw+ystart-50),fontFace=2, fontScale=1, color=(0,255,0), thickness=2)
                    subimg_list.append(subimg)
                # else:
                #     cv2.rectangle(draw_img,(xbox_left+xstart, ytop_draw+ystart),(xbox_left+win_draw+xstart,ytop_draw+win_draw+ystart),(0,0,255),2)

                # Animiation
                # plt.subplot(211)
                # plt.imshow(draw_img)
                # plt.subplot(212)
                # plt.imshow(cv2.cvtColor(subimg, cv2.COLOR_HSV2RGB))
                # plt.gcf().set_size_inches(10, 16)
                # display.display(plt.gcf())
                # display.clear_output(wait=True)
    return draw_img, box_list, energy_list

def add_heat(heatmap, bbox_list, energy_list, tao):
    # Iterate through list of bboxes
    result = (1-tao) * np.copy(heatmap)
    for box, energy in zip(bbox_list, energy_list):
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        result[box[1][0]:box[1][1], box[0][0]:box[0][1]] += tao * energy

    # Return updated heatmap
    return result

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def draw_heatmap(img1, heatmap):
    img_large = np.copy(img1)
    img_small = cv2.resize(heatmap,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    img_large[:img_small.shape[0],-img_small.shape[1]:,0] = img_small*10
    img_large[:img_small.shape[0],-img_small.shape[1]:,1] = img_small*10
    img_large[:img_small.shape[0],-img_small.shape[1]:,2] = img_small*10

    return img_large

class process_video(object):

    def __init__(self, coords, svc, X_scaler, config, threshold):
        self.heatmap = None
        self.labels = None
        self.coords = coords
        self.svc = svc
        self.X_scaler = X_scaler
        self.config = config
        self.threshold = threshold

    def output_heatmap_labels(self):
        return self.heatmap, self.labels

    def process_step(self,img):
        if self.heatmap is None:
            self.heatmap = np.zeros_like(img[:,:,0])
        image_boxes, box_list, energy_list = find_cars_multi_scales(img, self.coords, self.svc, self.X_scaler, self.config)
        current_heatmap = add_heat(self.heatmap, box_list, energy_list, 1.0)
        filtered_heatmap = add_heat(self.heatmap, box_list, energy_list, 0.5)
        threshold_heatmap = apply_threshold(filtered_heatmap, self.threshold)
        labels = label(threshold_heatmap)
        draw_img = draw_labeled_bboxes(np.copy(image_boxes), labels)
        # draw_img = draw_heatmap(draw_img, threshold_heatmap.astype(np.uint8))

        self.heatmap = np.copy(filtered_heatmap)
        self.labels  = labels
        # draw_img[current_heatmap>0,:] = np.int_(0.8*draw_img[current_heatmap>0,:]+[51,0,0])
        return draw_img
