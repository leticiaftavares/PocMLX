//
//  FlowerClassifier.swift
//  FlowersMLX
//
//  Created by Mateus Rodrigues on 12/08/26.
//

import MLX
import MLXNN

final class FlowerClassifier: Module {

    @ModuleInfo var conv1: Conv2d
    @ModuleInfo var conv2: Conv2d
    @ModuleInfo var conv3: Conv2d
    @ModuleInfo var conv4: Conv2d

    @ModuleInfo var dense1: Linear
    @ModuleInfo var dropout: Dropout
    @ModuleInfo var output: Linear

    override init() {

        // 128x128x3 -> 128x128x16
        _conv1 = ModuleInfo(wrappedValue: Conv2d(
            inputChannels: 3,
            outputChannels: 16,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
            bias: true
        ))

        // 64x64x16 -> 64x64x32
        _conv2 = ModuleInfo(wrappedValue: Conv2d(
            inputChannels: 16,
            outputChannels: 32,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
            bias: true
        ))

        // 32x32x32 -> 32x32x64
        _conv3 = ModuleInfo(wrappedValue: Conv2d(
            inputChannels: 32,
            outputChannels: 64,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
            bias: true
        ))

        // 16x16x64 -> 16x16x128
        _conv4 = ModuleInfo(wrappedValue: Conv2d(
            inputChannels: 64,
            outputChannels: 128,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
            bias: true
        ))

        // Global average pooling gives 128 features.
        _dense1 = ModuleInfo(wrappedValue: Linear(
            128,
            64
        ))

        _dropout = ModuleInfo(wrappedValue: Dropout(
            p: 0.2
        ))

        // Five flower classes.
        _output = ModuleInfo(wrappedValue: Linear(
            64,
            5
        ))

        super.init()
    }

    func callAsFunction(
        _ input: MLXArray
    ) -> MLXArray {

        var x = input

        x = relu(conv1(x))
        x = MaxPool2d(
            kernelSize: 2,
            stride: 2
        )(x)

        x = relu(conv2(x))
        x = MaxPool2d(
            kernelSize: 2,
            stride: 2
        )(x)

        x = relu(conv3(x))
        x = MaxPool2d(
            kernelSize: 2,
            stride: 2
        )(x)

        x = relu(conv4(x))

        // Global average pooling.
        //
        // [batch, height, width, 128]
        //              ↓
        // [batch, 128]
        x = mean(
            x,
            axes: [1, 2]
        )

        x = relu(dense1(x))

        x = dropout(x)

        // Raw logits.
        return output(x)
    }
}