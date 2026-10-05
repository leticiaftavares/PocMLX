final class FlowerCNN: Module {
    let features: Sequential
    let classifier: Sequential
    override init() {
        features = Sequential([
            Conv2d(
                inputChannels: 3,
                outputChannels: 64,
                kernelSize: 5,
                padding: 2
            ),
            ReLU(),
            MaxPool2d(
                kernelSize: 2,
                stride: 2
            ),
            Conv2d(
                inputChannels: 64,
                outputChannels: 64,
                kernelSize: 3,
                padding: 1
            ),
            ReLU(),
            MaxPool2d(
                kernelSize: 2,
                stride: 2
            ),
            Conv2d(
                inputChannels: 64,
                outputChannels: 64,
                kernelSize: 3,
                padding: 1
            ),
            ReLU(),
            MaxPool2d(
                kernelSize: 2,
                stride: 2
            ),
            Conv2d(
                inputChannels: 64,
                outputChannels: 64,
                kernelSize: 3,
                padding: 1
            ),
            ReLU(),
            MaxPool2d(
                kernelSize: 2,
                stride: 2
            )
        ])
        classifier = Sequential([
            Linear(
                inputDimensions: 14 * 14 * 64,
                outputDimensions: 512
            ),
            ReLU(),
            // 2 classes
            Linear(
                inputDimensions: 512,
                outputDimensions: 2
            )
        ])
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = features(x)
        // NHWC: [batch, 14, 14, 64]
        // Flatten to: [batch, 14 * 14 * 64]
        x = x.reshaped([x.shape[0], 14 * 14 * 64])
        return classifier(x)
    }
}