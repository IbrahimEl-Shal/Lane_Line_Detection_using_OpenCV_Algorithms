# Finding Lane Lines on the Road with OpenCV

The lane markers are obvious to any human observer. We perform processing of this image intuitively, and after being trained to drive a human can detect the lane in which the vehicle appears to be moving. Humans also effortlessly identify many other objects in the scene, such as the other vehicles, the embankment near the right shoulder, some road signs alongside the road, and even the mountains visible on the horizon. While many of these objects are complex in visual structure, it could be said that the lane markers are actually some of the simplest structures in the image!

Pre-existing knowledge of driving gives us certain assumptions about the properties and structure of a lane, further simplifying the problem.

The following steps show that what we did in this project.
  - Make a pipeline that finds lane lines on the road.
  - Apply our pipeline on test images.
  - Enhance the draw function to create a single line to represent of each line group.
  - Write a video processing pipeline.

# Reflection

My pipeline consisted of 5 steps. 
1) Converted the images to grayscale and apply gaussian smoothing filter.
2) Apply Canny Algorthim to detecting edges in the our test images.
3) Defining the Region of Interest which it should be a lines in front of cars.
4) Generating lines from edge pixels using Hough Transform.
5) Draw the lines on the edge image.

If you'd like to include images to show how the pipeline works, here is how to include an image:
<figure>
 <img src="examples/line-segments-example.jpg" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> First output after detecting line segments </p> 
 </figcaption>
</figure>
 <p></p> 
<figure>
 <img src="examples/laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Our enhancement after connect and average line segments to get output like this</p> 
 </figcaption>
</figure>

In order to draw a single line on the left and right lanes, I modified the draw_lines() function to Enhanced_Draw_Lines by the difference between the two groups of line segments is the direction of their slope and average the lines in each group into a single line that fits pretty closely in orientation and location in the image.

# Conclusion
Also,this project was successful in that the video images clearly show the lane lines are detected properly and lines are very smoothly handled. 

It only detects the straight lane lines. It is an advanced topic to handle curved lanes (or the curvature of lanes). We'll need to use perspective transformation and also poly fitting lane lines rather than fitting to straight lines.

Having said that, the lanes near the car are mostly straight in the images. The curvature appears at further distance unless it's a steep curve. So, this basic lane finding technique is still very useful.

Another thing is that it won't work for steep (up or down) roads because the region of interest mask is assumed from the center of the image.

For steep roads, we first need to detect the horizontal line (between the sky and the earth) so that we can tell up to where the lines should extend.