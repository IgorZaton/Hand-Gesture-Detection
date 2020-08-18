# Hand Gesture Detection
This project is convolutional neural network that can predict hand gests. I've collected data using get_data() function for 4 cases: open hand, fist, peace and yolo gests.
The CNN is created with tensorflow keras. Train data are available in data directiory. Those are grey-scale pictures 37x50 .png.
Input pictures are basically camera video with 10 shots per second.
All API stuff was created with tkinter.

When You run the program You have to set the mask range, so the input images would be nice and smooth. I've applied this so everyone could use it in every light conditions etc.
When the mask is set You can press Start button and CNN will start predict. Results of predictions are display in simple tkinter window with progress bars.

![](https://github.com/IgorZaton/Hand-Gesture-Detection/blob/master/maskrange.png)

![](https://github.com/IgorZaton/Hand-Gesture-Detection/blob/master/mask.png)
![](https://github.com/IgorZaton/Hand-Gesture-Detection/blob/master/predictions.png)
