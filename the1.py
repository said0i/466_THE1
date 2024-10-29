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
    #return upsampled_image
    '''
    #cropping
    #crop unnecessary parts due to rotation
    cos=abs(np.cos(np.radians(degree)))
    sin=abs(np.sin(np.radians(degree)))
    sum=cos+sin
    actual_width = (new_height*sin - new_width*cos) / (sin**2 - cos**2)
    actual_height = (new_width*sin - new_height*cos) / (sin**2 - cos**2)

    wfloor = int(((new_width - actual_width) + 2)// 2)
    wceil = int((new_width + actual_width) // 2 )
    hfloor = int(((new_height - actual_height) + 2)// 2)
    hceil = int((new_height + actual_height) // 2 )
    cropped_image = upsampled_image[wfloor:wceil,hfloor:hceil]
    return cropped_image
    '''
def compute_distance(img1, img2):
    distance = np.mean((img1 - img2) ** 2)
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
    kl_divergence = np.sum(np.where(hist1 != 0, (hist1/pixel_count) * np.log(hist1 / (hist2 + 1e-9)), 0))
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
    '''
    desert1_hist = desert1_hist / np.sum(desert1_hist)
    desert2_hist = desert2_hist / np.sum(desert2_hist)
    forest1_hist = forest1_hist / np.sum(forest1_hist)
    forest2_hist = forest2_hist / np.sum(forest2_hist)
    '''
    kl_div_deserts = compute_kl_divergence(desert1_hist, desert2_hist)
    kl_div_forests = compute_kl_divergence(forest1_hist, forest2_hist)
    kl_div_desert_forest1 = compute_kl_divergence(desert1_hist, forest1_hist)
    kl_div_desert_forest2 = compute_kl_divergence(desert2_hist, forest1_hist)
    kl_div_desert_forest3 = compute_kl_divergence(desert1_hist, forest2_hist)
    kl_div_desert_forest4 = compute_kl_divergence(desert2_hist, forest2_hist)
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

    return result

def difference_images(img1, img2):
    '''img1 and img2 are the images to take dhe difference
    returns the masked image'''
    #masked_image = cv2.absdiff(img2, img1)

    # Compute the absolute difference directly
    diff = cv2.absdiff(img1, img2)
    #create a mask where different pixels are 1 and others are 0
    mask = np.where(diff > 75, 1, 0)
    #apply the mask
    masked_image = mask * img2
    return masked_image

if __name__ == '__main__':
    '''
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
    #print('The distance between original image and image corrected with linear interpolation is ', compute_distance(img_original, corrected_img_linear))
    #print('The distance between original image and image corrected with cubic interpolation is ', compute_distance(img_original, corrected_img_cubic))

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
    '''
    '''
    ###################### Q2
    img = read_image('q2_1.jpg')
    result = desert_or_forest(img)
    print("Given image q2_1 is an image of a ", result)

    img = read_image('q2_2.jpg')
    result = desert_or_forest(img)
    print("Given image q2_2 is an image of a ", result)

    '''
    ###################### Q3
    img1 = read_image('q3_a1.png',gray_scale=True)
    img2 = read_image('q3_a2.png',gray_scale=True)
    result = difference_images(img1,img2)
    write_image(result, 'masked_image_a.png')

    img1 = read_image('q3_b1.png')
    img2 = read_image('q3_b2.png')
    result = difference_images(img1,img2)
    write_image(result, 'masked_image_b.png')



