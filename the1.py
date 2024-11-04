import numpy as np
import cv2 
import matplotlib.pyplot as  plt

input_folder = 'THE1_Images/'
output_folder = 'THE1_Outputs/'


def read_image(filename, gray_scale = False):
    # CV2 is just a suggestion you can use other libraries as well
    if gray_scale:
        img = cv2.imread(input_folder + filename, cv2.IMREAD_GRAYSCALE)    
        return img
    img = cv2.imread(input_folder + filename)
    return img

def write_image(img, filename):
    # CV2 is just a suggestion you can use other libraries as well
    cv2.imwrite(output_folder+filename, img)

def rotate_upsample(img, scale, degree, interpolation_type):
    '''img: img to be rotated and upsampled
    scale: scale of upsampling (e.g. if current width and height is 64x64, and scale is 4, wodth and height of the output should be 256x256)
    degree: shows the degree of rotation
    interp: either linear or cubic'''
    width = img.shape[1]
    height = img.shape[0]
    center = (width//2, height//2)
    M = cv2.getRotationMatrix2D(center, -degree, 1)
    aligned_image = cv2.warpAffine(img, M, (width,height))

    
    #upsampling

    # Get the new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    if(interpolation_type == 'linear'):
        # Upsample the image using INTER_LINEAR (linear interpolation)
        upsampled_image = cv2.resize(aligned_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    else:
        # Upsample the image using INTER_CUBIC (higher quality for enlarging)
        upsampled_image = cv2.resize(aligned_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return upsampled_image

def compute_distance(original, corrected):
    h1, w1, _ = original.shape
    h2, w2, _ = corrected.shape
    # The size of the corrected image is bigger due to black outer region coming from rotation.
    # In order to calculate distance, we have to make the sizes of the images equal.
    corrected = corrected[(h2-h1)//2:(h2+h1)//2 , (w2-w1)//2:(w2+w1)//2]
    distance = np.mean((original - corrected) ** 2)
    return distance

def rgb_to_hsi(image):
    # Convert to float to prevent overflow issues
    image = image.astype(np.float32) / 255
    R, G, B = image[:, :, 2], image[:, :, 1], image[:, :, 0]
    I = (R + G + B) / 3

    # Saturation calculation
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = 1 - (3 / (R + G + B + 1e-6)) * min_rgb  # Add epsilon to avoid division by zero

    # Hue calculation
    numerator = 0.5 * ((R - G) + (R - B))
    denominator = np.sqrt((R - G)**2 + (R - B) * (G - B)) + 1e-6  # Avoid division by zero
    theta = np.arccos(numerator / denominator)
    theta=np.degrees(theta)
    H = np.where(B <= G, theta, 360 - theta)
    #H = H / (2 * np.pi)  # Normalize to [0,1]

    # Stack H, S, I channels into an HSI image
    hsi_image = np.stack((H, S, I), axis=-1)
    return hsi_image

def compute_kl_divergence(hist1, hist2):
    pixel_count= hist1.sum()
    # Only compute log for non-zero entries in hist1
    kl_divergence = np.sum(np.where(hist1 != 0, (hist1/pixel_count) * np.log((hist1 + 1e-10) / (hist2 + 1e-10)), 0))
    return kl_divergence

def desert_or_forest(img):
    '''img: image to be classified as desert or forest
    return a string: either 'desert'  or 'forest' 
    
    You should compare the KL Divergence between histograms of hue channel. Please provide images and discuss these histograms in your report'''
    desert1 = read_image('desert1.jpg')
    desert2 = read_image('desert2.jpg')
    forest1 = read_image('forest1.jpg')
    forest2 = read_image('forest2.jpg')

    desert1_hsi = rgb_to_hsi(desert1)
    desert2_hsi = rgb_to_hsi(desert2)
    forest1_hsi = rgb_to_hsi(forest1)
    forest2_hsi = rgb_to_hsi(forest2)
    img_hsi = rgb_to_hsi(img)
    # Compute histograms for the Hue channel
    bins = 360  # Number of bins  
    hist_range = (0, 360)  # For normalized HSI Hue range
    def compute_histogram(image):
        h_channel = image[:, :, 0]  # Hue channel
        hist = cv2.calcHist([h_channel], [0], None, [bins], hist_range)
        #cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_)
        return hist

    desert1_hist = compute_histogram(desert1_hsi)
    desert2_hist = compute_histogram(desert2_hsi)
    forest1_hist = compute_histogram(forest1_hsi)
    forest2_hist = compute_histogram(forest2_hsi)
    img_hist=compute_histogram(img_hsi)
    # Plot and save each histogram individually
    plt.figure()
    plt.plot(desert1_hist, label='Desert 1')
    plt.title('Desert 1 Hue Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(output_folder + 'desert1_histogram.png')
    plt.close()

    plt.figure()
    plt.plot(desert2_hist, label='Desert 2')
    plt.title('Desert 2 Hue Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(output_folder + 'desert2_histogram.png')
    plt.close()

    plt.figure()
    plt.plot(forest1_hist, label='Forest 1')
    plt.title('Forest 1 Hue Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(output_folder + 'forest1_histogram.png')
    plt.close()

    plt.figure()
    plt.plot(forest2_hist, label='Forest 2')
    plt.title('Forest 2 Hue Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(output_folder + 'forest2_histogram.png')
    plt.close()
    
    
    # Compute KL Divergence between the histograms
    
    kl_div_desert1 = compute_kl_divergence(desert1_hist, img_hist)
    kl_div_desert2 = compute_kl_divergence(desert2_hist, img_hist)
    kl_div_forest1 = compute_kl_divergence(forest1_hist, img_hist)
    kl_div_forest2 = compute_kl_divergence(forest2_hist, img_hist)

    # Compare the KL Divergence values
    if(kl_div_desert1+kl_div_desert2 < kl_div_forest1+kl_div_forest2):
        result = 'desert'
    else:
        result = 'forest'

    plt.figure()
    plt.plot(img_hist, label=result + '_input')
    plt.title(result + '_input ' + 'Hue Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(output_folder + result + '_input'+ '_histogram.png')
    plt.close()

    return result

def difference_images(img1, img2):
    '''img1 and img2 are the images to take dhe difference
    returns the masked image'''
    if(len(img1.shape) == 2):#grayscale
        # Compute the absolute difference directly
        diff = cv2.absdiff(img1, img2)
        '''
        #create a mask where different pixels are 1 and others are 0
        mask = np.where(diff > 75, 1, 0)
        #apply the mask
        masked_image = (mask * img2).astype(np.uint8)
        ret,thresh = cv2.threshold(diff,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #ret,thresholding=cv2.threshold(diff, 0, 255,cv2.THRESH_BINARY)
        #return thresh
        return masked_image
        '''
        diff = diff.astype(np.uint8)
        # Use Otsu's method on the difference to get an adaptive threshold
        ret, k = cv2.threshold(diff, 0, 255, cv2.THRESH_OTSU)
        # Create a mask where the significant differences (above the threshold) are kept
        mask = np.where(diff > ret, 1, 0).astype(np.uint8)
        
        # Apply the mask to one of the original images to extract the object
        masked_image = (mask * img2).astype(np.uint8)
        inverse_masked_image = 255-masked_image
        ret2, thresh = cv2.threshold(inverse_masked_image,0, 255, cv2.THRESH_OTSU)
        mask2 = np.where(inverse_masked_image < ret2, 1, 0).astype(np.uint8)
        
        final_image = (mask2 * img2).astype(np.uint8)
        return k

    else:#rgb
        # Compute the absolute difference for each channel
        diff_r = cv2.absdiff(img1[:, :, 2], img2[:, :, 2])
        diff_g = cv2.absdiff(img1[:, :, 1], img2[:, :, 1])
        diff_b = cv2.absdiff(img1[:, :, 0], img2[:, :, 0])
        # Create a mask where different pixels are 1 and others are 0
        mask = np.where((diff_r > 25) | (diff_g > 25) | (diff_b > 25), 1, 0)
        # Apply the mask
        masked_image = np.stack((mask * img2[:, :, 2], mask * img2[:, :, 1], mask * img2[:, :, 0]), axis=-1)
        return masked_image

if __name__ == '__main__':
    ###################### Q1
    # Read original image
    img_original = read_image('q1_1.png')
    # Read corrupted image
    img = read_image('ratio_4_degree_30.png')
    # Correct the image with linear interpolation
    corrected_img_linear = rotate_upsample(img, 4, 30, 'linear')
    write_image(corrected_img_linear, 'q1_1_corrected_linear.png')
    # Correct the image with cubic interpolation
    corrected_img_cubic = rotate_upsample(img, 4, 30, 'cubic')
    write_image(corrected_img_cubic, 'q1_1_corrected_cubic.png')

    # Report the distances
    print('The distance between original image and image corrected with linear interpolation is ', compute_distance(img_original, corrected_img_linear))
    print('The distance between original image and image corrected with cubic interpolation is ', compute_distance(img_original, corrected_img_cubic))

    # Repeat the same steps for the second image
    img_original = read_image('q1_2.png')
    img = read_image('ratio_8_degree_45.png')
    corrected_img_linear = rotate_upsample(img, 8, 45, 'linear')
    write_image(corrected_img_linear, 'q1_2_corrected_linear.png')
    corrected_img_cubic = rotate_upsample(img, 8, 45, 'cubic')
    write_image(corrected_img_cubic, 'q1_2_corrected_cubic.png')

    # Report the distances
    print('The distance between original image and image corrected with linear interpolation is ', compute_distance(img_original, corrected_img_linear))
    print('The distance between original image and image corrected with cubic interpolation is ', compute_distance(img_original, corrected_img_cubic))
    ###################### Q2
    img = read_image('q2_1.jpg')
    result = desert_or_forest(img)
    print("Given image q2_1 is an image of a ", result)

    img = read_image('q2_2.jpg')
    result = desert_or_forest(img)
    print("Given image q2_2 is an image of a ", result)

    
    ###################### Q3
    img1 = read_image('q3_a1.png',gray_scale=True)
    img2 = read_image('q3_a2.png',gray_scale=True)
    result = difference_images(img1,img2)
    write_image(result, 'masked_image_a.png')

    img1 = read_image('q3_b1.png')
    img2 = read_image('q3_b2.png')
    result = difference_images(img1,img2)
    write_image(result, 'masked_image_b.png')



