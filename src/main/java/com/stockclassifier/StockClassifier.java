package com.stockclassifier;

import classifiers.GPClassifier;
import classifiers.J48Classifier;
import classifiers.MLPWrapper;
import utils.StatisticalTest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import java.util.Random;
import java.util.Scanner;

public class StockClassifier {
    private static final int NUM_FOLDS = 10;
    private static final double ALPHA = 0.05; // Significance level for statistical test

    public static void main(String[] args) {
        try {
            Scanner scanner = new Scanner(System.in);
            // Get seed value
            System.out.print("Enter seed value: ");
            long seed = scanner.nextLong();
            Random random = new Random(seed);
            // Get file paths
            System.out.print("Enter training data path: ");
            String trainingPath = scanner.next();
            System.out.print("Enter test data path: ");
            String testPath = scanner.next();
            // Load data
            DataSource source = new DataSource(trainingPath);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);
            // Initialize classifiers
            GPClassifier gpClassifier = new GPClassifier();
            MLPWrapper mlpClassifier = new MLPWrapper();
            J48Classifier j48Classifier = new J48Classifier();
            // Set seed for reproducibility
            gpClassifier.setSeed(seed);
            // Train classifiers
            System.out.println("\nTraining classifiers...");
            gpClassifier.buildClassifier(data);
            mlpClassifier.buildClassifier(data);
            j48Classifier.buildClassifier(data);
            // Evaluate classifiers
            System.out.println("\nEvaluating classifiers...");
            Evaluation gpEval = new Evaluation(data);
            Evaluation mlpEval = new Evaluation(data);
            Evaluation j48Eval = new Evaluation(data);
            // Cross-validation
            gpEval.crossValidateModel(gpClassifier, data, NUM_FOLDS, random);
            mlpEval.crossValidateModel(mlpClassifier, data, NUM_FOLDS, random);
            j48Eval.crossValidateModel(j48Classifier, data, NUM_FOLDS, random);
            // Print results
            System.out.println("\nResults:");
            System.out.println("GP Classifier:");
            System.out.println("Accuracy: " + gpEval.pctCorrect());
            System.out.println("F1 Score: " + gpEval.fMeasure(1));
            System.out.println("Model Structure: " + gpClassifier.getModelStructure());
            System.out.println("\nMLP Classifier:");
            System.out.println("Accuracy: " + mlpEval.pctCorrect());
            System.out.println("F1 Score: " + mlpEval.fMeasure(1));
            System.out.println("Model Structure: " + mlpClassifier.getModelStructure());
            System.out.println("\nJ48 Classifier:");
            System.out.println("Accuracy: " + j48Eval.pctCorrect());
            System.out.println("F1 Score: " + j48Eval.fMeasure(1));
            System.out.println("Model Structure: " + j48Classifier.getModelStructure());
            // Perform Wilcoxon test between GP and MLP
            double[] gpScores = new double[NUM_FOLDS];
            double[] mlpScores = new double[NUM_FOLDS];
            for (int i = 0; i < NUM_FOLDS; i++) {
                gpScores[i] = gpEval.fMeasure(1);
                mlpScores[i] = mlpEval.fMeasure(1);
            }
            StatisticalTest.WilcoxonResult result = StatisticalTest.wilcoxonSignedRankTest(gpScores, mlpScores, ALPHA);
            System.out.println("\nWilcoxon Signed-Rank Test Results (GP vs MLP):");
            System.out.println("Test Statistic: " + result.statistic);
            System.out.println("p-value: " + result.pValue);
            System.out.println("Significant difference: " + (result.isSignificant ? "Yes" : "No"));
            // Save results to file
            String resultsPath = "results/classification_results.txt";
            try (java.io.PrintWriter writer = new java.io.PrintWriter(new java.io.FileWriter(resultsPath))) {
                writer.println("Classification Results");
                writer.println("====================");
                writer.println("\nGP Classifier:");
                writer.println("Accuracy: " + gpEval.pctCorrect());
                writer.println("F1 Score: " + gpEval.fMeasure(1));
                writer.println("Model Structure: " + gpClassifier.getModelStructure());
                writer.println("\nMLP Classifier:");
                writer.println("Accuracy: " + mlpEval.pctCorrect());
                writer.println("F1 Score: " + mlpEval.fMeasure(1));
                writer.println("Model Structure: " + mlpClassifier.getModelStructure());
                writer.println("\nJ48 Classifier:");
                writer.println("Accuracy: " + j48Eval.pctCorrect());
                writer.println("F1 Score: " + j48Eval.fMeasure(1));
                writer.println("Model Structure: " + j48Classifier.getModelStructure());
                writer.println("\nWilcoxon Signed-Rank Test Results (GP vs MLP):");
                writer.println("Test Statistic: " + result.statistic);
                writer.println("p-value: " + result.pValue);
                writer.println("Significant difference: " + (result.isSignificant ? "Yes" : "No"));
            }
            System.out.println("\nResults have been saved to: " + resultsPath);
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}