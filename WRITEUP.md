# Project Write-Up
This write-up helps show my understanding of the OpenVINO™ Toolkit and its impact on performance, as well as articulating the use cases of the great application I have deployed at the edge.

## Explaining Custom Layers
Custom layers are a necessary and important to have feature of the OpenVINO™ Toolkit.   Although we should not have to use it very often, if at all, due to all of the supported layers. However, it is useful to know a little about its existence and how to use it if the need arises.  Custom layers are those outside of the list of known, supported layers, and are typically a rare exception. Handling custom layers in a neural network for use with the Model Optimizer depends somewhat on the framework used; other than adding the custom layer as an extension, we have to follow instructions specific to the framework.

### The process behind converting custom layers 
involves the model optimizer and the inference engine.  First the model optimizer searches the list of known layers for each layer in the input model. The inference engine loads the layers from the input model IR files into the specified device plugin, which will search a list of known layer implementations for the device. If the model architecure contains layer or layers that are not in the list of known layers for the device, the Inference Engine considers the layer to be unsupported and reports an error. 

To implement a custom layer for a pre-trained model in the Open Vino toolkit, we add extensions to both the Model Optimizer and the Inference Engine.  Here, I'll explain what the model optimizer and the inference engine do.

### Basic Processing Steps of the Model Optimizer
The Model Optimizer extracts information from the input model which includes the topology of the model layers along with parameters, input and output format for each layer. The model is optimized from the various known characteristics of the layers, interconnects, and data flow which partly comes from the layer operation providing details including the shape of the output for each layer. Then the optimized model is output to the model IR files needed by the Inference Engine to run the model.

* There are two custom layer extensions required:

### Custom Layer Extractor:
Identifies the custom layer operation and extracts the parameters for each instance of the custom layer. The layer parameters are stored per instance and used by the layer operation before finally appearing in the output IR. Typically the input layer parameters are unchanged, which is the case covered by this tutorial.

### Custom Layer Operation:
Specifies the attributes that are supported by the custom layer and computes the output shape for each instance of the custom layer from its parameters. The --mo-op command-line argument shown in the examples below generates a custom layer operation for the Model Optimizer.

### The Inference Engine
Within the inference engine, each device plugin includes a library of optimized implementations to execute known layer operations which must be extended to execute a custom layer. The custom layer extension is implemented according to the target device:

### Custom Layer CPU Extension:
is a compiled shared library (.so or .dll binary) needed by the CPU Plugin for executing the custom layer on the CPU.

### Custom Layer GPU Extension:
Consists of the OpenCL source code (.cl) for the custom layer kernel that will be compiled to execute on the GPU along with a layer description file (.xml) needed by the GPU Plugin for the custom layer kernel.

### Why Use Model Extension Generator?

The Model Extension Generator tool generates template source code files for each of the extensions needed by the Model Optimizer and the Inference Engine.

The script for this is available in the Command-line:
/opt/intel/openvino/deployment_tools/tools/extension_generator/extgen.py

The script can be used in the following manner:

usage: Any combination of the following arguments can be used:

Arguments to configure extension generation in the interactive mode:

optional arguments:
  -h, --help            show this help message and exit
  --mo-caffe-ext        generate a Model Optimizer Caffe* extractor
  --mo-mxnet-ext        generate a Model Optimizer MXNet* extractor
  --mo-tf-ext           generate a Model Optimizer TensorFlow* extractor
  --mo-op               generate a Model Optimizer operation
  --ie-cpu-ext          generate an Inference Engine CPU extension
  --ie-gpu-ext          generate an Inference Engine GPU extension
  --output_dir OUTPUT_DIR
                        set an output directory. If not specified, the current
                        directory is used by default.


### Some of the potential reasons for handling custom layers are:
To actually add custom layers, there are a few differences depending on the original model framework. In both TensorFlow and Caffe, the first option is to register the custom layers as extensions to the Model Optimizer.

For Caffe, the second option is to register the layers as Custom, then use Caffe to calculate the output shape of the layer. You’ll need Caffe on your system to do this option.

For TensorFlow, its second option is to actually replace the unsupported subgraph with a different subgraph. The final TensorFlow option is to actually offload the computation of the subgraph back to TensorFlow during inference.


In addition, one common use case would be when using lambda layers. These layers allow to add an arbitrary peice of code to your model implementation. In order to have support for these kind of layers, we use custom layers. 

And it's also very important to be able to convert custom layers because many developers might be developing something new or researching on something and for the application to work smoothly, we need to know how the usefulness and support for custom layers.


## Comparing Model Performance:

The SSD MobileNet V2 COCO model worked well with this application so I ended up using it. Here is how it compares to the two other models that did not with this APP:

### Model size

                        SSD MobileNet V2 COCO	 ---- SSD Inception V2 COCO ---- SSD MobileNet V1 COCO
Before Conversion	----      69.7 MB ----                   103 MB  ----            29 MB
After Conversion	----      65 MB	  ----                     97 MB ----            27 MB

### Inference Time

                        SSD MobileNet V2 COCO ---- SSD Inception V2 COCO ---- SSD MobileNet V1 COCO
Before Conversion	----    50 ms -----                  150 ms ----                 55 ms
After Conversion	----    60 ms -----                  155 ms ----                 60 ms



## Assess Model Use Cases

Some of the potential use cases of the people counter app are:

People counters
Monitors space
Security system
Optimize staff placements
Optimize personnels
Queue managemnet techniques
Increase customer retention
Space management in stores, workplace
Discover when stores generate the most traffic which might lead to sales
Calculate percentage of people who leave stores due to long queues
In Airports
Identify which shops perform well
Managing long lines.

The people counters use case could count people as they walk into a facility or zone. It then separates the counts by entrance and by time, adding in variables such as sales transactions, daily special events, weather conditions, and staffing hours. With this key data, a full suite of business intelligence reports are generated, enabling store owners and business managers to make informed business decisions.

This application could also help in making best use of spaces in public areas. It can also help monitor space and get instantly notified of a tailgating incident before it becomes a security threat to your team in realtime.  It could asist in learning more about your space utilization while keeping privacy top of mind within your organization.  Airports are one of the busiest places and with so much data with an application like this, some wonderful insights could be derived. 

Each of these use cases would be useful because at some point, we all might have experienced leaving a store due to long lines and a similar application could help in resolving this issue. This application can also be used in security systems.  The system could identify if a person has broken in a shop and in the case capture a photo or send alerts. This could be an automated and effective solution to thefts.  


## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are as follows:  I'll start with camera focal length/image size have different effects on a deployed edge model.

Distorted input from camera due to change in focal length and/or image size will affect the model because the model may fail to make sense of the input and the distored input may not be detected properly by the model. An approach to solve this would be to use some augmnted images while training models and specifying the threshold skews, this could be a potential solution. However, the ranges would need to be selected correctly or it could lead to a loss of accuracy.

Natural decrease in model accuracy during conversion or other stages may make the model unusable if it doesn't perform the required task such as detecting risk of crimes as mentioned above in use-cases. An effectuve solution to this would be to put the model into validation mode for an hour or so. During this time, the various devices could perform federated learning to give better performance. This might drastically provide a better performance.

In case of poor lighting, model's accuracy may fail dramatically or may even completely drop close to zero. However, this can be mitigated with good hardware that can process the images from poorly lit regions before passing it to the model.





