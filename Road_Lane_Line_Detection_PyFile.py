# In[1]:Import Packages

import os
import cv2
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import HTML
from moviepy.editor import VideoFileClip # Import everything needed to edit/save/watch video clips

get_ipython().run_line_magic('matplotlib', 'inline')

# In[2]:Functions Implementation

def grayscale(img):

    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):

    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):

    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):

    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    
    plt.imshow(masked_image)
    plt.show()
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (int(x1),int(y1)), (int(x2), int(y2)), color, thickness)
            
def Enhanced_Draw_Lines(img, lines, color=[255, 0, 0], thickness=2):

    RLane_x = []
    RLane_y = []
    LLane_x = []
    LLane_y = []
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            Slope = (y2-y1) / (x2-x1)
            if (math.fabs(Slope) < 0.5):  
                continue
            if Slope <= 0: 
                LLane_x.extend([x1, x2])
                LLane_y.extend([y1, y2])
            else: 
                RLane_x.extend([x1, x2])
                RLane_y.extend([y1, y2])
                
    min_y = img.shape[0] * (3 / 5) # <-- Just below the horizon
    max_y = img.shape[0] # <-- The bottom of the image

    poly_left = np.poly1d(np.polyfit(LLane_y,LLane_x, deg=1))    
    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))
    
    poly_right = np.poly1d(np.polyfit(RLane_y,RLane_x,deg=1))
    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))

    line_image = draw_lines(img,[[
                                    [left_x_start, max_y, left_x_end, min_y],
                                    [right_x_start, max_y, right_x_end, min_y],
                                ]],
                            thickness = 5)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #draw_lines(line_img, lines)  
    Enhanced_Draw_Lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):

    return cv2.addWeighted(initial_img, α, img, β, γ)

def Load_Images(Path_Directory):
    Images_in_Path = []
    for filename in os.listdir(Path_Directory):
        ext = os.path.splitext(filename)[1]
        if ext.lower() not in Valid_images:
            continue
        Images_in_Path.append(Image.open(os.path.join(Path_Directory,filename))) 
        
    return (Images_in_Path)

def Save_Image(Image_Number,Picture):
    
    save_path = 'test_images_output/'
    if not os.path.exists(save_path):
                os.makedirs(save_path)
    
    filename = "image%d.png" % Image_Number             
    mpimg.imsave(os.path.join(save_path,filename),Picture)
    
    return (True)
    
def Process_Image(Imgs):
    
    ROT_Vertices = [(480, 300),(110,540),(880, 540)] #[Vertices,RHS(X,Y),LHS(X,Y)]
    Color_Select = np.copy(Imgs)  
        
    # printing out some stats and plotting
    print('This image is:', type(Color_Select), 'with dimesions:', Color_Select.shape) 
    
    Gray = grayscale(Color_Select)        #Apply Grayscale the image
    Blur_Gray = gaussian_blur(Gray, 3)    #Apply Gaussian smoothing
    Edges = canny(Blur_Gray, 50, 150)     #Apply Canny Algorthim
    
    # Defining a four sided polygon to mask
    Masked_Image = region_of_interest(Edges, np.array([ROT_Vertices], np.int32))
    
    #Run Hough on edge detected image
    Lines = hough_lines(Masked_Image, HTP['rho'], HTP['theta'], 
                        HTP['threshold'], HTP['min_line_len'], HTP['max_line_gap'])    

    # Draw the lines on the edge image
    Lines_Edges = weighted_img(Lines, Color_Select)

    plt.imshow(Lines_Edges)
    plt.show()

    return (Lines_Edges)

# In[3]:Build a Lane Finding Pipeline

HTP = {#Define Hough transform parameters
       'rho':1,            #Distance resolution in pixels of the Hough grid
       'theta':np.pi/180,  #Angular resolution in radians of the Hough grid# 
       'threshold':30,     #Minimum number of votes (intersections in Hough grid cell)
       'min_line_len':50,  #Minimum number of pixels making up a line 150 - 40
       'max_line_gap':150, #Maximum gap in pixels between connectable line segments 
       }

Valid_images = [".jpg",".gif",".png",".tga"]

Imgs = Load_Images("test_images/")

for Pic in range(0,len(Imgs)): 
    Get_Test_Image = Process_Image(Imgs[Pic])
    Save_Image(Pic,Get_Test_Image)
        
# In[4]: Test solution on provided videos

white_output = 'test_videos_output/solidWhiteRight_init.mp4'
Video_1 = VideoFileClip('test_videos/solidWhiteRight.mp4')
Video_1.reader.close()
Video_1.audio.reader.close_proc()
white_clip = Video_1.fl_image(Process_Image).subclip(0,3)#NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

HTML("""<video width="960" height="540" controls><source src="{0}"></video>""".format(white_output))

# In[5]: Test solution on provided videos

yellow_output = 'test_videos_output/solidYellowLeft.mp4'
Video_2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
Video_2.reader.close()
Video_2.audio.reader.close_proc()
yellow_clip = Video_2.fl_image(Process_Image)
yellow_clip.write_videofile(yellow_output, audio=False)