package com.tylerapo;
import java.io.*; 
import weka.core.*; 
import weka.core.Instances; 
import weka.classifiers.Evaluation; 
import weka.classifiers.functions.MultilayerPerceptron; 
import weka.core.converters.ArffLoader;

public class NeuralNetworkDemo {
	
	public static Instances getDataSet(String fileName) throws IOException {
		/**
		 * we can set the file i.e., loader.setFile("filename") to load the data
		 */
		int classIdx = 1;
		/** the arffloader to load the arff file */
		ArffLoader loader = new ArffLoader();
		/** load the traing data */
		loader.setSource(KNNDemo.class.getResourceAsStream("./" + fileName));
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
	
	NeuralNetworkDemo(){ 
		try{ 
			
			Instances train = getDataSet("contact-lenses.arff"); 
			Instances test = getDataSet("contact-lenses-test.arff");
			train.setClassIndex(train.numAttributes() - 1); 
			test.setClassIndex(test.numAttributes() - 1); 
		
			MultilayerPerceptron mlp = new MultilayerPerceptron(); 
			mlp.setOptions(Utils.splitOptions("-L 0.3 -M 0.2 -N 500 -V 0 -S 0 -E 20 -H 4")); 
		
			mlp.buildClassifier(train); 
		
			Evaluation eval = new Evaluation(train); 
			eval.evaluateModel(mlp, test); 
			System.out.println(eval.toSummaryString("\nResults\n======\n", false)); 
 
		} catch(Exception ex){ 
			ex.printStackTrace(); 
		} 
	} 

	public static void main(String args[]){ 
		new NeuralNetworkDemo(); 
	} 
} 