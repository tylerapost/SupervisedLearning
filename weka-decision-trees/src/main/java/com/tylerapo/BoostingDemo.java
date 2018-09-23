package com.tylerapo;
import java.io.*;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.NBTree;
import weka.classifiers.trees.RandomForest;
import weka.core.FastVector;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader;

public class BoostingDemo {
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
	   

	   public static void main(String[] args) throws Exception {
		    Instances train = getDataSet("vote.arff");
	
		    Instances test = getDataSet("voteTest.arff");
		    train.setClassIndex(train.numAttributes() - 1);
	        test.setClassIndex(test.numAttributes()-1);
	
	        Classifier [] ClassifierArray=new Classifier[3];
	        ClassifierArray[1]=new J48();
	        ClassifierArray[0]=new NaiveBayes();
	        ClassifierArray[2]=new NBTree();
	        CVParameterSelection ps = new CVParameterSelection();
	        AdaBoostM1 vs=new AdaBoostM1();
	        //String[] options=new String[3];
	        //options[2]="-R MAJ";
	        //options[1]="-B weka.classifiers.functions.SMO -B weka.classifiers.bayes.NaiveBayes";
	        //options[0]="-S <2>";
	        //vs.setOptions(options);
	        ps.setOptions(weka.core.Utils.splitOptions("-P \"I 1.0 10.0 1.0\" -P \"P 1.0 100.0 10.0\" -W \"weka.classifiers.meta.AdaBoostM1\" -- -P 100 -S 1 -I 10 -W \"weka.classifiers.functions.SMO\" -- -V -1 -C 1 -P 1.0E-12"));
	
	        ps.buildClassifier(train);
	
	        System.out.println(Utils.joinOptions(ps.getBestClassifierOptions()));
	        String[] options=ps.getBestClassifierOptions();
	        //SET options of adaboost to our chosen classifier
	        vs.setOptions(options);
	        System.out.println(Utils.joinOptions(vs.getOptions()));
	        vs.setClassifier(ClassifierArray[2]);
	
	        vs.buildClassifier(train);
	        //find optimal parameter
	
	        //Dagging cls = new Dagging();
	        //change the base classifier
	        //cls.setClassifier(new NBTree());
	        //change the parameter for dagging
	        //cls.setNumFolds(1);
	        //cls.setSeed(7);
	        //cls.buildClassifier(train);
	        //System.out.println(((Object) vs).getCombinationRule());
	        System.out.println(vs.getOptions());
	        PrintWriter pw=new PrintWriter(new FileWriter("boostresults.txt"));
	
	        //System.out.println(Utils.joinOptions(ps.getBestClassifierOptions()));
	        for (int i = 0; i < test.numInstances(); i++) {
	            double pred = vs.classifyInstance(test.instance(i));
	            pw.println(pred);
	        }
	        pw.close();
	        //weka.core.SerializationHelper.write("/Weka-3-6/ProjectMilestone3/ionosphere.model", vs);
	        Evaluation eval=new Evaluation(train);
	        eval.evaluateModel(vs,test);
	        System.out.println("** Boosting Example **");
			System.out.println(eval.toSummaryString());
			System.out.print(" the expression for the input data as per alogorithm is ");
			System.out.println(eval.toMatrixString());
			System.out.println(eval.toClassDetailsString());
	        Double error_c=eval.errorRate();
	        System.out.println(error_c);
	   }
}

