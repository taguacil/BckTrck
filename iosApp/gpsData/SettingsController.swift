//
//  SettingsController.swift
//  gpsData
//
//  Created by Taimir Aguacil on 14.12.18.
//  Copyright © 2018 Taimir Aguacil. All rights reserved.
//

import UIKit
import CoreLocation
import os.log

extension UITextField {
    func addDoneCancelToolbar(onDone: (target: Any, action: Selector)? = nil, onCancel: (target: Any, action: Selector)? = nil) {
        let onCancel = onCancel ?? (target: self, action: #selector(cancelButtonTapped))
        let onDone = onDone ?? (target: self, action: #selector(doneButtonTapped))
        
        let toolbar: UIToolbar = UIToolbar()
        toolbar.barStyle = .default
        toolbar.items = [
            UIBarButtonItem(title: "Cancel", style: .plain, target: onCancel.target, action: onCancel.action),
            UIBarButtonItem(barButtonSystemItem: .flexibleSpace, target: self, action: nil),
            UIBarButtonItem(title: "Done", style: .done, target: onDone.target, action: onDone.action)
        ]
        toolbar.sizeToFit()
        
        self.inputAccessoryView = toolbar
    }
    
    // Default actions:
    @objc func doneButtonTapped() { self.resignFirstResponder() }
    @objc func cancelButtonTapped() { self.text = ""
        self.resignFirstResponder() }
}

class SettingsController: UIViewController, UITextFieldDelegate, UINavigationControllerDelegate {
    
    // MARK: Properties
    var iterations:Int?
    var pathLength:Int?
    var samplingRatio:Double?
    var learningRate:Float?
    var locationVector : [CLLocation]?
    var est_coord : [CLLocationCoordinate2D]?
    var AvgMSE : Int?
    
    let alert = UIAlertController(title: "Computing", message: "please wait", preferredStyle: UIAlertController.Style.alert)
    let alertParams = UIAlertController(title: "Invalid parameters!", message: "Parameters not correct, cannot proceed", preferredStyle: UIAlertController.Style.alert)
    /*let progressBar = UIProgressView(progressViewStyle: .default)*/
    
    @IBOutlet weak var iterTextField: UITextField! {
        didSet { iterTextField?.addDoneCancelToolbar() }
    }
    @IBOutlet weak var pathLengthTextField: UITextField!
        {
        didSet { pathLengthTextField?.addDoneCancelToolbar() }
    }
    @IBOutlet weak var learningRateTextField: UITextField!
        {
        didSet { learningRateTextField?.addDoneCancelToolbar() }
    }
    @IBOutlet weak var samplingRatioTextField: UITextField!
        {
        didSet { samplingRatioTextField?.addDoneCancelToolbar() }
    }
    
    @IBOutlet weak var applyButton: UIButton!
    
    @IBOutlet weak var progressBar:UIProgressView!
    @IBOutlet weak var progressLabel: UILabel!
    @IBOutlet weak var mseLabel: UILabel!
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        iterTextField.delegate = self
        pathLengthTextField.delegate = self
        learningRateTextField.delegate = self
        samplingRatioTextField.delegate = self
        
        // Enable the Apply button only if all fields have valid content.
        updateSaveButtonState()
        //progressBar.frame = CGRect(x: 10, y: 70, width: 250, height: 0)
        init_variable_params()
        //alert.view.addSubview(progressBar)
        alertParams.addAction(UIAlertAction(title: "Continue", style: UIAlertAction.Style.default, handler:nil))
    }
    
    //MARK: UITextFieldDelegate
    @objc func onCancelTextField(textField:UITextField) {
        textField.text=""
        textField.resignFirstResponder()
    }
    
    func textFieldShouldReturn(_ textField: UITextField) -> Bool {
        // Hide the keyboard
        textField.resignFirstResponder()
        return true
    }
    
    func textFieldDidBeginEditing(_ textField: UITextField) {
        updateSaveButtonState()
    }
    
    func textFieldDidEndEditing(_ textField: UITextField) {
        updateSaveButtonState()
    }
    
    // MARK: - Navigation
    /*@IBAction func cancelButton(_ sender: Any) {
     let tmpController :UIViewController! = self.presentingViewController
     
     self.dismiss(animated: true, completion: {()->Void in
     tmpController.dismiss(animated: true, completion: nil)
     })
     }*/
    
    @IBAction func applyButton(_ sender: UIButton) {
        //DispatchQueue.main.async {
        init_variable_params()
        self.progressLabel.textColor = UIColor.blue
        self.mseLabel.textColor = UIColor.blue
        
        let startDate = Date()
        
        if self.updateParams()
        {
            //self.present(alert, animated: true, completion: nil)
            // Do the time critical stuff asynchronously
            DispatchQueue.global(qos: .default).async {
                
                if let CS = CompressSensing(inputLocationVector: self.locationVector!)
                {
                    os_log("Computation starts...", log: OSLog.default, type: .debug)
                    CS.setParam(maxIter: self.iterations!, pathLength: self.pathLength!, samplingRatio: self.samplingRatio!, learningRate: self.learningRate! )
                    let (est_coord, AvgMSE) = CS.compute(obj:self, date:startDate)
                    self.est_coord = est_coord
                    self.AvgMSE = AvgMSE
                }
                DispatchQueue.main.async {
                    /*self.alert.dismiss(animated: true, completion: {self.performSegue(withIdentifier: "showReconstruct", sender: nil)})*/
                    self.performSegue(withIdentifier: "showReconstruct", sender: nil)
                }
            }
        }
        else
        {
            os_log("Parameters not correct, cannot proceed", log: OSLog.default, type: .error)
            self.present(alertParams, animated: true, completion: nil)
        }
        //    }
    }
    
    @IBAction func setDefault(_ sender: UIButton) {
        iterTextField.text="512"
        pathLengthTextField.text = "256"
        samplingRatioTextField.text = "0.2"
        learningRateTextField.text = "0.01"
        updateSaveButtonState()
    }
    
    
    // This method lets you configure a view controller before it's presented.
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        super.prepare(for : segue , sender: sender)
        // Configure the destination view controller only when the save button is pressed.
        /*guard let button = sender as? UIButton, button === applyButton else {
         os_log("The apply button was not pressed, cancelling", log: OSLog.default, type: .debug)
         return
         }*/
        
        let alert = UIAlertController(title: "Invalid parameters", message: "One or more parameters are not set correctly", preferredStyle: UIAlertController.Style.alert)
        alert.addAction(UIAlertAction(title: "Back", style: UIAlertAction.Style.default, handler: nil))
        
        switch(segue.identifier ?? "") {
        case "showReconstruct":
            guard let routeViewController = segue.destination as? RouteViewController else {
                fatalError("Unexpected destination: \(segue.destination)")
            }
            
            routeViewController.locationVector = locationVector
            sleep(2)
            routeViewController.est_coord = est_coord
            routeViewController.AvgMSE = AvgMSE
        default:
            fatalError("Unexpected Segue Identifier; \(String(describing: segue.identifier))")
        }
    }
    
    
    //MARK: Private Methods
    private func updateSaveButtonState() {
        // Disable the Save button if the text field is empty.
        let text = iterTextField.text ?? ""
        applyButton.isEnabled = !text.isEmpty
    }
    
    private func init_variable_params(){
        progressBar.setProgress(0.0, animated: true)
        progressLabel.text = String(format: "0 secs")
        mseLabel.text = String(format: "0 meters")
        progressLabel.textColor = UIColor.black
        mseLabel.textColor = UIColor.black
    }
    
    private func updateParams()->Bool {
        // Make some checks on params and assign them
        iterations = Int(iterTextField.text!)
        pathLength = Int(pathLengthTextField.text!)
        samplingRatio = Double(samplingRatioTextField.text!)
        learningRate = Float(learningRateTextField.text!)
        
        guard (iterations! > 0), (iterations! < 2000) else
        {
            os_log("Number of iteration out of bound", log: OSLog.default, type: .debug)
            return false
        }
        guard ((pathLength! % 2)==0) else
        {
            os_log("Invalid path length (Power of 2 for DCT)", log: OSLog.default, type: .debug)
            return false
        }
        guard (samplingRatio!>0), (samplingRatio!<=1) else
        {
            os_log("Invalid sampling ratio", log: OSLog.default, type: .debug)
            return false
        }
        guard (learningRate!>0) else {
            os_log("Invalid learning rate", log: OSLog.default, type: .debug)
            return false
        }
        return true
    }
}
