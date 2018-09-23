package com.tylerapo;

import java.io.IOException;
import java.text.DecimalFormat;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

// found at: https://www.programcreek.com/java-api-examples/?api=weka.classifiers.functions.LibSVM

public class SVMWekaDemo {
	public static Instances getDataSet(String fileName) throws IOException {
		/**
		 * we can set the file i.e., loader.setFile("filename") to load the data
		 */
		int classIdx = 1;
		/** the arffloader to load the arff file */
		ArffLoader loader = new ArffLoader();
		/** load the traing data */
		loader.setSource(SVMWekaDemo.class.getResourceAsStream("./" + fileName));
		/**
		 * we can also set the file like loader3.setFile(new
		 * File("test-confused.arff"));
		 */
		//loader.setFile(new File(fileName));
		Instances dataSet = loader.getDataSet();
		/** set the index based on the data given in the arff files */
		dataSet.setClassIndex(classIdx);
		return dataSet;
	}
	public static void main(String[] args) throws Exception {

		SMO wekaClassifier = new SMO();
		wekaClassifier.setOptions(new String[] {"-B", "-H"});

		Instances preparedData = getDataSet("cpu.arff");
		Instances preparedTest = getDataSet("cpuTest.arff");
		
		System.out.println("Reading train set and test set done!");

		System.out.print("\nTraining...");
		wekaClassifier.buildClassifier(preparedData);
		
		System.out.println("\nTraining...done!");
		
		Evaluation evalTrain = new Evaluation(preparedData);
		evalTrain.evaluateModel(wekaClassifier, preparedData);

		DecimalFormat formatter = new DecimalFormat("#0.0");
		
		System.out.println("\nEvaluating on trainSet...");
		System.out.println(evalTrain.toSummaryString());
		
		System.out.println("\nResult on trainSet...");
		System.out.println("Precision:" + formatter.format(100*evalTrain.precision(0)) + "%" +
				" - Recal: " + formatter.format(100*evalTrain.recall(0)) + "%" +
				" - F1: " + formatter.format(evalTrain.fMeasure(0)) + "%");
		
		Evaluation eval = new Evaluation(preparedTest);
		eval.evaluateModel(wekaClassifier, preparedTest);

		System.out.println("\nEvaluating on testSet...");
		System.out.println(eval.toSummaryString());
		
		System.out.println("\nResult on testSet...");
		System.out.println("Precision:" + formatter.format(100*eval.precision(0)) + "%" +
				" - Recal: " + formatter.format(100*eval.recall(0)) + "%" +
				" - F1: " + formatter.format(100*eval.fMeasure(0)) + "%");

		System.out.println("True positive rate: " + formatter.format(100*eval.truePositiveRate(0)) + "%" + 
				" - True negative rate: "  + formatter.format(100*eval.trueNegativeRate(0)) + "%");
		System.out.println("Accuracy: " + formatter.format(100*((eval.truePositiveRate(0) + eval.trueNegativeRate(0)) / 2)) + "%");
		
		System.out.println("\nDone!");
	}
}
