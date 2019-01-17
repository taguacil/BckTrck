//
//  CompressSensing.swift
//  gpsData
//
//  Created by Taimir Aguacil on 17.08.18.
//  Copyright © 2018 Taimir Aguacil. All rights reserved.
//

/*
 Init with location vector
 Downsample by random sampling
 Transform to ValueArray
 Normalize
 Compute DCT of eye(N) and pick on N*ratio or comupte directly
 Perform Lasso
 Get the weights
 Perform IDCT
 Compute MSE
 Revert the normalization
 Send back the reconstructed signal
 Plot both paths and display accuracy in %
 */

import Accelerate
import Upsurge
import MachineLearningKit
import CoreLocation
import os.log
import CoreML

extension Array {
    /// Picks `n` random elements (partial Fisher-Yates shuffle approach)
    subscript (randomPick n: Int) -> [Element] {
        var copy = self
        for i in stride(from: count - 1, to: count - n - 1, by: -1) {
            copy.swapAt(i, Int(arc4random_uniform(UInt32(i + 1))))
        }
        return Array(copy.suffix(n))
    }
}

class CompressSensing : NSObject, NSCoding {
    
    //MARK: Properties
    let nnModel = OptimalNet1()
    
    var totalEstimate : TimeInterval
    let locationVector : [CLLocation]?
    var iteration : Int?
    var ratio : Double?
    var l1_penalty : Float? // learning rate
    var blockLength : Int?
    var NNCompute : Bool?
    var LASSOCompute : Bool?
    
    var progress = Float(0)
    
    var weights_lat: Matrix<Float>!
    var weights_lon: Matrix<Float>!
    
    var latArray_org: [Float]
    var lonArray_org : [Float]
    var lat_est : [Float]
    var lon_est : [Float]
    var latValArray : Array<Float>
    var lonValArray : Array<Float>
    
    var latNN_est = Array<Double>()
    var lonNN_est = Array<Double>()
    var latArrayNN_org: [Double]
    var lonArrayNN_org : [Double]
    
    
    var meanLat = Float(0)
    var meanLon = Float(0)
    var stdLat  = Float(0)
    var stdLon  = Float(0)
    
    lazy var dctSetupForward: vDSP_DFT_Setup = {
        guard let setup = vDSP_DCT_CreateSetup(
            nil,
            vDSP_Length(blockLength!),
            .II)else {
                fatalError("can't create forward vDSP_DFT_Setup")
        }
        return setup
    }()
    
    lazy var dctSetupInverse: vDSP_DFT_Setup = {
        guard let setup = vDSP_DCT_CreateSetup(
            nil,
            vDSP_Length(blockLength!),
            .III) else {
                fatalError("can't create inverse vDSP_DFT_Setup")
        }
        
        return setup
    }()
    
    var blockSamples : Int
    let tolerance = Float(0.0001)
    let lassModel = LassoRegression()
    
    //MARK: Archiving Paths
    
    static let DocumentsDirectory = FileManager().urls(for: .documentDirectory, in: .userDomainMask).first!
    static let ArchiveURL = DocumentsDirectory.appendingPathComponent("locationVector")
    
    //MARK: Types
    struct PropertyKey {
        static let locationVector = "locationVector"
    }
    
    //Mark: Initializer
    init?(inputLocationVector: [CLLocation]) {
        
        self.blockSamples = 0
        self.latArray_org = []
        self.lonArray_org = []
        self.lat_est = []
        self.lon_est = []
        self.latValArray = []
        self.lonValArray = []
        
        self.latArrayNN_org = []
        self.lonArrayNN_org = []
        
        self.totalEstimate = 0
        
        // Extract the coordinates from the location vector if it exists
        guard !inputLocationVector.isEmpty else {
            os_log("No data", log: OSLog.default, type: .debug)
            return nil
        }
        self.locationVector = inputLocationVector
        //self.latValArray = Array<Float> (repeating:0, count: Int(blockSamples))
        //self.lonValArray = Array<Float> (repeating:0, count: Int(blockSamples))
        
    }
    
    //MARK: DeInit
    deinit {
        vDSP_DFT_DestroySetup(dctSetupForward)
        vDSP_DFT_DestroySetup(dctSetupInverse)
    }
    
    //MARK: NSCoding
    func encode(with aCoder: NSCoder) {
        aCoder.encode(locationVector, forKey: PropertyKey.locationVector)
    }
    
    required convenience init?(coder aDecoder: NSCoder) {
        // The name is required. If we cannot decode a name string, the initializer should fail.
        guard let locationVector = aDecoder.decodeObject(forKey: PropertyKey.locationVector) as? [CLLocation] else {
            os_log("Unable to decode the locationVector for a CompressSensing object.", log: OSLog.default, type: .debug)
            return nil
        }
        // Must call designated initializer.
        self.init(inputLocationVector: locationVector)
    }
    
    
    //MARK: RandomSampling and conversion to 2 Upsurge vectors
    private func randomSampling() -> [Int]{
        os_log("Random sampling", log: OSLog.default, type: .debug)
        let indices = Array(0...blockLength!-1)
        
        var downSampledIndices = indices[randomPick: blockSamples]
        downSampledIndices  = downSampledIndices.sorted()
        
        // Put indices element in upsurge vectors
        for item in downSampledIndices {
            latValArray.append(latArray_org[item])
            lonValArray.append(lonArray_org[item])
        }
        meanLat = mean(latValArray)
        meanLon = mean(lonValArray)
        stdLat  = std(latValArray)
        stdLon  = std(lonValArray)
        if stdLat == 0 { stdLat = 1}
        if stdLon == 0 { stdLon = 1}
        latValArray = Array((latValArray-meanLat)/(stdLat))
        lonValArray = Array((lonValArray-meanLon)/(stdLon))
        return downSampledIndices
    }
    
    //MARK: DCT operations
    private func forwardDCT<M: LinearType>(_ input: M) -> [Float] where M.Element == Float {
        os_log("Forward DCT", log: OSLog.default, type: .debug)
        var results = Array<Float>(repeating:0, count: Int(blockLength!))
        let realVector = ValueArray<Float>(input)
        vDSP_DCT_Execute(dctSetupForward,
                         realVector.pointer,
                         &results)
        return results
    }
    
    /* Performs a real to read forward IDCT  */
    private func inverseDCT<M: LinearType>(_ input: M) -> [Float] where M.Element == Float {
        os_log("Inverse DCT", log: OSLog.default, type: .debug)
        var results = Array<Float>(repeating:0, count: Int(blockLength!))
        let realVector = ValueArray<Float>(input)
        vDSP_DCT_Execute(dctSetupInverse,
                         realVector.pointer,
                         &results)
        return results
    }
    
    //MARK: DCT of Identity
    private func eyeDCT(downSampledIndices: [Int]) -> [Array<Float>] {
        os_log("Identiy DCT function", log: OSLog.default, type: .debug)
        var pulseVector = Array<Float>(repeating: 0.0, count: blockLength!)
        let dctVec = ValueArray<Float>(capacity: blockSamples*blockLength!)
        var dctMat = Matrix<Float>(rows: blockSamples, columns: blockLength!)
        var dctArrayMat = Array<Array<Float>>() // TODO not efficient
        for item in downSampledIndices {
            pulseVector[item] = 1.0
            let dctVector = forwardDCT(pulseVector)
            pulseVector[item] = 0.0
            dctVec.append(contentsOf: dctVector)
        }
        dctMat = dctVec.toMatrix(rows: blockSamples, columns: blockLength!)
        
        /* print debugging
         print(dctVec.description)
         print(dctMat.description)
         */
        
        var temp = Array<Float>()
        for col in 0..<dctMat.columns
        {
            for row in 0..<dctMat.rows
            {
                temp.append(dctMat[row, col])
            }
            dctArrayMat.append(temp)
            temp.removeAll(keepingCapacity: true)
        }
        return dctArrayMat
    }
    
    //MARK: LASSO regression
    private func lassoReg (dctMat : [Array<Float>]){
        os_log("Lasso regression function", log: OSLog.default, type: .debug)
        // Set Initial Weights
        let initial_weights_lat = Matrix<Float>(rows: blockLength!+1, columns: 1, repeatedValue: 0)
        let initial_weights_lon = Matrix<Float>(rows: blockLength!+1, columns: 1, repeatedValue: 0)
        weights_lat = try! lassModel.train(dctMat, output: latValArray, initialWeights: initial_weights_lat, l1Penalty: l1_penalty!, tolerance: tolerance, iteration : iteration!)
        
        /* print debugging
         print(dctMat.description)
         print(latValArray.description)
         print(weights_lat.description)
         print(weights_lat.column(1).description)
         */
        
        weights_lon = try! lassModel.train(dctMat, output: lonValArray, initialWeights: initial_weights_lon, l1Penalty: l1_penalty!, tolerance: tolerance, iteration : iteration!)
    }
    
    /* Performs IDCT of weights and renormalizes*/
    private func IDCT_weights(downSampledIndices: [Int]) {
        os_log("IDCT of weights", log: OSLog.default, type: .debug)
        var lat_cor = inverseDCT(weights_lat.column(1))
        var lon_cor = inverseDCT(weights_lon.column(1))
        lat_cor = Array(lat_cor*(1/Float(sqrt(ratio!*0.5*Double(blockLength!)))))
        lon_cor = Array(lon_cor*(1/Float(sqrt(ratio!*0.5*Double(blockLength!)))))
        renorm(downSampledIndices: downSampledIndices, lat_cor: lat_cor, lon_cor: lon_cor)
    }
    
    /* Renormalization */
    private func renorm(downSampledIndices: [Int], lat_cor: [Float], lon_cor: [Float]){
        os_log("Renormalization LASSO", log: OSLog.default, type: .debug)
        var vec_lat = Array<Float>()
        var vec_lon = Array<Float>()
        
        for index in 0..<downSampledIndices.count
        {
            vec_lat.append(latValArray[index]-lat_cor[downSampledIndices[index]])
            vec_lon.append(lonValArray[index]-lon_cor[downSampledIndices[index]])
        }
        let delta_lat = mean(vec_lat)
        let delta_lon = mean(vec_lon)
        
        lat_est = Array((lat_cor+delta_lat)*stdLat+meanLat)
        lon_est = Array((lon_cor+delta_lon)*stdLon+meanLon)
    }
    
    /* Renormalization NN */
    private func renormNN(downSampledIndices: [Int]){
        os_log("Renormalization NN", log: OSLog.default, type: .debug)
        var vec_lat = Array<Double>()
        var vec_lon = Array<Double>()
        
        for index in 0..<downSampledIndices.count
        {
            vec_lat.append(Double(latValArray[index])-latNN_est[downSampledIndices[index]])
            vec_lon.append(Double(lonValArray[index])-lonNN_est[downSampledIndices[index]])
        }
        let delta_lat = mean(vec_lat)
        let delta_lon = mean(vec_lon)
        
        latNN_est = Array((latNN_est+delta_lat)*Double(stdLat)+Double(meanLat))
        lonNN_est = Array((lonNN_est+delta_lon)*Double(stdLon)+Double(meanLon))
    }
    
    
    /* MSE */
    private func MSE(lat_est:Array<Float>, lon_est:Array<Float>, lat_org: Array<Float>, lon_org: Array<Float>, latNN_est:Array<Double>, lonNN_est:Array<Double>, latNN_org: Array<Double>, lonNN_org: Array<Double>) -> (Float,Float) {
        let MSE = sqrt(measq((lat_est-lat_org))+measq((lon_est-lon_org)))
        let MSEnn = sqrt(measq((latNN_est-latNN_org))+measq((lonNN_est-lonNN_org)))
        print("Total latlon MSE \(MSE), \(MSEnn)")
        return (MSE, Float(MSEnn))
    }
    //MARK: Complete computation for 1 block length
    private func computeBlock() -> () {
        let downSampledIndices = randomSampling()
        // LASSO
        if (LASSOCompute!)
        {
            let dctMat = eyeDCT(downSampledIndices: downSampledIndices)
            lassoReg(dctMat: dctMat)
            IDCT_weights(downSampledIndices: downSampledIndices)
        }
        // NN
        if (NNCompute!)
        {
            guard let latNNout = try? nnModel.prediction(input: OptimalNet1Input(input1:preprocess(Array: latValArray)!)) else {
                fatalError("Unexpected runtime error.")
            }
            latNN_est = postprocess(NNout: latNNout.output1)
            guard let lonNNout = try? nnModel.prediction(input: OptimalNet1Input(input1:preprocess(Array: lonValArray)!)) else {
                fatalError("Unexpected runtime error.")
            }
            lonNN_est = postprocess(NNout: lonNNout.output1)
            
            // Renormalize both lat/lon for NN
            renormNN(downSampledIndices: downSampledIndices)
        }
    }
    
    // UI update
    private func updateProgress(obj:SettingsController, diff: TimeInterval){
        print (progress)
        obj.progressBar.setProgress(progress, animated: true)
        //let prog_per = progress*100.0
        //obj.progressLabel.text = String(format: "%.1f%%", prog_per)
        obj.progressLabel.text = String(format: "%.1f secs", diff)
    }
    
    // Preprocess and prepare input for NeuralNetworks
    private func preprocess(Array: [Float]) -> MLMultiArray? {
        guard let array = try? MLMultiArray(shape: [51 as NSNumber], dataType: .double) else {
            return nil
        }
        for (index, element) in Array.enumerated() {
            array[index] = NSNumber(value: Double(element))
        }
        return array
    }
    
    // Postprocess NN output
    private func postprocess(NNout: MLMultiArray) -> [Double] {
        let length = NNout.count
        let doublePtr =  NNout.dataPointer.bindMemory(to: Double.self, capacity: length)
        let doubleBuffer = UnsafeBufferPointer(start: doublePtr, count: length)
        let output = Array(doubleBuffer)
        return output
    }
    
    // Entire computation for all input vector
    func compute(obj:SettingsController, date:Date) -> ([CLLocationCoordinate2D], [CLLocationCoordinate2D],Int, Int) {
        let totalLength = locationVector!.count
        let numberOfBlocks = Int(floor(Double(totalLength / blockLength!)))
        var latTotal_est = Array<Float>()
        var lonTotal_est = Array<Float>()
        var latTotal_org = Array<Float>()
        var lonTotal_org = Array<Float>()
        var est_coord = [CLLocationCoordinate2D]()
        var est_coordNN = [CLLocationCoordinate2D]()
        
        var latTotalNN_est = Array<Double>()
        var lonTotalNN_est = Array<Double>()
        var latTotalNN_org = Array<Double>()
        var lonTotalNN_org = Array<Double>()
        
        for i in 0..<numberOfBlocks
        {
            let startIteration = Date()
            latArray_org.removeAll()
            lonArray_org.removeAll()
            lat_est.removeAll()
            lon_est.removeAll()
            latValArray.removeAll()
            lonValArray.removeAll()
            
            latArrayNN_org.removeAll()
            lonArrayNN_org.removeAll()
            latNN_est.removeAll()
            lonNN_est.removeAll()
            
            for item in locationVector![i*(blockLength!)...((i+1)*blockLength!)-1] {
                latArray_org.append(Float(item.coordinate.latitude))
                lonArray_org.append(Float(item.coordinate.longitude))
                
                latArrayNN_org.append(Double(item.coordinate.latitude))
                lonArrayNN_org.append(Double(item.coordinate.longitude))
            }
            computeBlock()
            
            latTotal_est.append(contentsOf: lat_est)
            lonTotal_est.append(contentsOf: lon_est)
            latTotal_org.append(contentsOf: latArray_org)
            lonTotal_org.append(contentsOf: lonArray_org)
            
            latTotalNN_est.append(contentsOf: latNN_est)
            lonTotalNN_est.append(contentsOf: lonNN_est)
            latTotalNN_org.append(contentsOf: latArrayNN_org)
            lonTotalNN_org.append(contentsOf: lonArrayNN_org)
            
            for k in 0..<blockLength!
            {
                if (LASSOCompute!)
                {
                    est_coord.append(CLLocationCoordinate2DMake(Double(lat_est[k]), Double(lon_est[k])))
                }
                else
                {
                    est_coord.append(CLLocationCoordinate2DMake(0,0))
                }
                if (NNCompute!)
                {
                    est_coordNN.append(CLLocationCoordinate2DMake(latNN_est[k], lonNN_est[k]))
                }
                else
                {
                    est_coordNN.append(CLLocationCoordinate2DMake(0.0, 0.0))
                }
                
            }
            progress += 1.0/Float(numberOfBlocks)
            
            let diff = Date().timeIntervalSince(startIteration)
            //let diff = Date().timeIntervalSince(date)
            
            totalEstimate = Double(numberOfBlocks-i-1)*diff
            DispatchQueue.main.async{
                self.updateProgress(obj:obj, diff:self.totalEstimate)
            }
        }
        
        let (mse,mseNN) = MSE(lat_est: latTotal_est, lon_est: lonTotal_est, lat_org: latTotal_org, lon_org: lonTotal_org,latNN_est: latTotalNN_est, lonNN_est: lonTotalNN_est, latNN_org: latTotalNN_org, lonNN_org: lonTotalNN_org)
        
        var MSE_wm = 0
        if (LASSOCompute!)
        {
            MSE_wm = Int(mse*1e5)
        }
        var MSENN_wm = 0
        if (NNCompute!)
        {
            MSENN_wm = Int(mseNN*1e5)
        }
        DispatchQueue.main.async{
            //self.updateProgress(obj:obj,diff:diff)
            obj.mseLabel.text = String(format: "%d meters, %d meters", MSE_wm, MSENN_wm)
        }
        return (est_coord, est_coordNN, MSE_wm, MSENN_wm)
    }
    //MARK: Function to set parameters
    func setParam(maxIter:Int, pathLength:Int, samplingRatio:Double, learningRate:Float, bLASSO:Bool, bNN:Bool){
        os_log("Setting algorithm parameters", log: OSLog.default, type: .debug)
        iteration = maxIter
        blockLength = pathLength
        ratio = samplingRatio
        l1_penalty = learningRate
        blockSamples = Int(floor(Double(blockLength!)*ratio!)) // to sample in a block
        LASSOCompute = bLASSO
        NNCompute = bNN
    }
    
}
