# YOLO系列 改进



### 说明：主要涉及yolo系列如何去进行改进，如yolov5添加注意力机制，更换loss，调整尺寸要求等。在改进完成后，会使用一些数据集进行测试，但是并不代表实际的性能优化结果，仅仅代表改进的实现是行的通的。如果需要通过改进来优化模型在具体数据集上面的表现，可以使用模型改进的实现去进行实际训练。



### yolo-v3

- yolov3-ghostdarknet
  - 使用yolov5文件中的yolov3.yaml文件为基础，将backbone中的conv与bottleneck替换为ghostconv与ghostbottleneck
  - 模型的参数量受益于ghostnet的轻量化，相较于yolov3的参数量缩减约60%
  - 使用模型训练trash数据集（过拟合）

