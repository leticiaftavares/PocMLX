// The Swift Programming Language
// https://docs.swift.org/swift-book

import Foundation
import MLX
import MLXRandom
import CoreImage
import CoreML
import CreateMLComponents


@main
struct MyExecutable {
    static func main() throws {
   
        var daisyImages: [URL] = []
        
        if let directoryURL = Bundle.module.url(forResource: "daisy", withExtension: nil) {
            let files = try FileManager.default.contentsOfDirectory(at: directoryURL, includingPropertiesForKeys: nil)
            daisyImages.append(contentsOf: files)
        }
        
        var dandelionImages: [URL] = []
        
        if let directoryURL = Bundle.module.url(forResource: "dandelion", withExtension: nil) {
            let files = try FileManager.default.contentsOfDirectory(at: directoryURL, includingPropertiesForKeys: nil)
            dandelionImages.append(contentsOf: files)
        }
        
        print("daisy:", daisyImages.count)
        print(dandelionImages.count)

        
        let daisyData = try daisyImages.map { url in
            let data = try Data(contentsOf: url)
            let array = MLXArray(data.dropFirst(16), [data.count - 16], type: UInt8.self)
            return array
        }
        
        let (daisyTrain, daisyTest) = daisyData.randomSplit(by: 0.8)
        let (dandelionTrain, dandelionTest) = daisyData.randomSplit(by: 0.8)
        print(daisyTrain.count, daisyTest.count)
        print(dandelionTrain.count, dandelionTest.count)
        
        var trainData = daisyTrain + dandelionTrain
        var testData = daisyTest + dandelionTest
        trainData.shuffle()
        testData.shuffle()
        
        print(trainData.count, testData.count)
        
        
    }
}
