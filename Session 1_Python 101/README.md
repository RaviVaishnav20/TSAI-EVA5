**Question 1: What are Channels and Kernels (according to EVA)?**

**Answer:** A layer could have multiple channels (or feature maps): an input layer has 3 channels if the inputs are RGB images. “channel” is usually used to describe the structure of a “layer”. Similarly, “kernel” is used to describe the structure of a “filter”.

The difference between filter and kernel is a bit tricky. Sometimes, they are used interchangeably, which could create confusions. Essentially, these two terms have subtle difference. A “Kernel” refers to a 2D array of weights. The term “filter” is for 3D structures of multiple kernels stacked together. For a 2D filter, filter is same as kernel. But for a 3D filter and most convolutions in deep learning, a filter is a collection of kernels. Each kernel is unique, emphasizing different aspects of the input channel.

With these concepts, the multi-channel convolution goes as the following. Each kernel is applied onto an input channel of the previous layer to generate one output channel. This is a kernel-wise process. We repeat such process for all kernels to generate multiple channels. Each of these channels are then summed together to form one single output channel. The following illustration should make the process clearer.
Here the input layer is a 5 x 5 x 3 matrix, with 3 channels. The filter is a 3 x 3 x 3 matrix. First, each of the kernels in the filter are applied to three channels in the input layer, separately. Three convolutions are performed, which result in 3 channels with size 3 x 3.

![Kernel vs Channel]"https://github.com/RaviVaishnav20/TSAI-EVA5/blob/master/Session%201_Python%20101/images/kernelVSchannel_2.gif"

Then these three channels are summed together (element-wise addition) to form one single channel (3 x 3 x 1). This channel is the result of convolution of the input layer (5 x 5 x 3 matrix) using a filter (3 x 3 x 3 matrix).

![Kernel vs Channel]"https://github.com/RaviVaishnav20/TSAI-EVA5/blob/master/Session%201_Python%20101/images/kernelVSchannel_3.gif"

Equivalently, we can think of this process as sliding a 3D filter matrix through the input layer. Notice that the input layer and the filter have the same depth (channel number = kernel number). The 3D filter moves only in 2-direction, height & width of the image (That’s why such operation is called as 2D convolution although a 3D filter is used to process 3D volumetric data). At each sliding position, we perform element-wise multiplication and addition, which results in a single number. In the example shown below, the sliding is performed at 5 positions horizontally and 5 positions vertically. Overall, we get a single output channel.

 

**Question 2: Why should we (nearly) always use 3x3 kernels?**

**Answer :** A convolution filter passes over all the pixels of the image in such a manner that, at a given time, we take 'dot product' of the convolution filter and the image pixels to get one final value output. We do this hoping that the weights (or values) in the convolution filter, when multiplied with corresponding image pixels, gives us a value that best represents those image pixels. We can think of each convolution filter as extracting some kind of feature from the image.

Therefore, convolutions are done usually keeping these two things in mind -

Most of the features in an image are usually local. Therefore, it makes sense to take few local pixels at once and apply convolutions.
Most of the features may be found in more than one place in an image. This means that it makes sense to use a single kernel all over the image, hoping to extract that feature in different parts of the image.

![Small kernel vs Large kernel]"https://github.com/RaviVaishnav20/TSAI-EVA5/blob/master/Session%201_Python%20101/images/Small_vs_Large_Kernel.PNG"

Based on the points listed in the above table and from experimentation, smaller kernel filter sizes are a popular choice over larger sizes.

Another question could be the preference for odd number filters or kernels over 2X2 or 4X4.

![2x2 kernel]"https://github.com/RaviVaishnav20/TSAI-EVA5/blob/master/Session%201_Python%20101/images/2x2kernel.jpg"

The explanation for that is that though we may use even sized filters, odd filters are preferable because if we were to consider the final output pixel (of next layer) that was obtained by convolving on the previous layer pixels, all the previous layer pixels would be symmetrically around the output pixel. Without this symmetry, we will have to account for distortions across the layers. This will happen due to the usage of an even sized kernel. Therefore, even sized kernel filters aren’t preferred.


**Question 3:**How many times to we need to perform 3x3 convolutions operations to reach close to 1x1 from 199x199 (type each layer output like 199x199 > 197x197...)

**Answer :**
`199*199 | 3*3 |197*197 -> 197*197 | 3*3 | 195*195 -> 195*195 | 3*3 | 193*193 -> 193*193 | 3*3 | 191*191 -> 191*191 | 3*3 | 189*189 -> 189*189 | 3*3 | 187*187 -> 187*187 | 3*3 | 185*185 ->185*185 | 3*3 | 183*183 -> 183*183 | 3*3 | 181*181 -> 181*181 | 3*3 | 179 *179 -> 179 *179 | 3*3 | 177 * 177-> 177*177 | 3*3 | 175*175 ->175*175 | 3*3 | 173*173 -> 173*173 | 3*3 | 171*171 -> 171*171 | 3*3 | 169 *169 -> 169 *169 | 3*3 | 167 * 167-> 167*167 | 3*3 | 165*165 ->165*165 | 3*3 | 163*163 -> 163*163 | 3*3 | 161*161 -> 161*161 | 3*3 | 159 *159 -> 159 *159 | 3*3 | 157 * 157->157*157 | 3*3 | 155*155 ->155*155 | 3*3 | 153*153 -> 153*153 | 3*3 | 151*151 -> 151*151 | 3*3 | 149 *149 -> 149 *149 | 3*3 | 147 * 147->147*147 | 3*3 | 145*145 ->145*145 | 3*3 | 143*143 -> 143*143 | 3*3 | 141*141 -> 141*141 | 3*3 | 139 *139 -> 139 *139 | 3*3 | 137 * 137-> 137*137 | 3*3 | 135*135 ->135*135 | 3*3 | 133*133 -> 133*133 | 3*3 | 131*131 -> 131*131 | 3*3 | 129 *129 -> 129 *129 | 3*3 | 127 * 127 -> 127*127 | 3*3 | 125*125 ->125*125 | 3*3 | 123*123 -> 123*123 | 3*3 | 121*121 -> 121*121 | 3*3 | 119 *119 -> 119 *119 | 3*3 | 117 * 117-> 117*117 | 3*3 | 115*115 ->115*115 | 3*3 | 113*113 -> 113*113 | 3*3 | 111*111 -> 111*111 | 3*3 | 109 *109 -> 109 *109 | 3*3 | 107 * 107 -> 107*107 | 3*3 | 105*105 ->105*105 | 3*3 | 103*103 -> 103*103 | 3*3 | 101*101 -> 101*101 | 3*3 | 99 *99 -> 99 *99 | 3*3 | 97 * 97-> 97*97 | 3*3 | 95*95 ->95*95 | 3*3 | 93*93 -> 93*93 | 3*3 | 91*91 -> 91*91 | 3*3 | 89 *89 -> 89 *89 | 3*3 | 87 * 87 -> 87*87 | 3*3 | 85*85 ->85*85 | 3*3 | 83*83 -> 83*83 | 3*3 | 81*81 -> 81*81 | 3*3 | 79 *79 -> 79 *79 | 3*3 | 77 * 77 -> 77*77 | 3*3 | 75*75 ->75*75 | 3*3 | 73*73 -> 73*73 | 3*3 | 71*71 -> 71*71 | 3*3 | 69 *69 -> 69 *69 | 3*3 | 67 * 67-> 67*67 | 3*3 | 65*65 ->65*65 | 3*3 | 63*63 -> 63*63 | 3*3 | 61*61 -> 61*61 | 3*3 | 59 *59 -> 59 *59 | 3*3 | 57 * 57 -> 57*57 | 3*3 | 55*55 ->55*55 | 3*3 | 53*53 -> 53*53 | 3*3 | 51*51 -> 51*51 | 3*3 | 49 *49 -> 49 *49 | 3*3 | 47 * 47-> 47*47 | 3*3 | 45*45 ->45*45 | 3*3 | 43*43 -> 43*43 | 3*3 | 41*41 -> 41*41 | 3*3 | 39 *39 -> 39 *39 | 3*3 | 37 * 37 -> 37*37 | 3*3 | 35*35 ->35*35 | 3*3 | 33*53 -> 33*33 | 3*3 | 31*31 -> 31*31 | 3*3 | 29 *29 -> 29 *29 | 3*3 | 27 * 27-> 27*27 | 3*3 | 25*25 ->25*25 | 3*3 | 23*23 -> 23*23 | 3*3 | 21*21 -> 21*21 | 3*3 | 19 *19 -> 19 *19 | 3*3 | 17 * 17 -> 17*17 | 3*3 | 15*15 ->15*15 | 3*3 | 13*13 -> 13*13 | 3*3 | 11*11 -> 11*11 | 3*3 | 09 *09 -> 09 *09 | 3*3 | 07 * 07-> 07*07 | 3*3 | 05*05 ->05*05 | 3*3 | 03*03 -> 03*03 | 3*3 | 01*01`

Total Number of Kernels used -> 1004

 

**Question 4: How are kernels initialized?**

**Answer :** Each kernel is initialized with a random value.

 

**Question 5:** What happens during the training of a DNN?

**Answer:** The filters at the first few initial layers extract direction & color, these primitive(edges & color) feature maps get combined into basic grids, textures & patterns. These, in turn, get combined to extract increasingly complex features which resemble parts of objects. As we move down the network the features extracted become more complex to interpret.

![CNN training]"https://github.com/RaviVaishnav20/TSAI-EVA5/blob/master/Session%201_Python%20101/images/CNN_training.png"

The above diagram shows feature maps at different stages on the network. The feature maps at the initial(lower) layers encode edges & direction i.e. horizontal or vertical lines, the feature maps obtained at the middle of the network visualize textures & patterns, the feature maps at top layers depict parts of objects & objects in the image.

