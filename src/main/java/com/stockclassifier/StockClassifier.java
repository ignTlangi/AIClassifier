package com.stockclassifier;

import classifiers.GPClassifier;
import classifiers.J48Classifier;
import classifiers.MLPWrapper;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import java.io.File;
import java.io.FileWriter;
import java.util.Random;
import java.util.Scanner;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import java.util.HashMap;
import java.util.Map;
import utils.StatisticalTest;
import java.util.InputMismatchException;

public class StockClassifier {
    private static final int NUM_FOLDS = 10;
    private static final double ALPHA = 0.05; // Significance level for statistical test
    private static final Map<String, double[]> resultsMap = new HashMap<>();

    public static void main(String[] args) {
        try {
            Scanner scanner = new Scanner(System.in);
            
            // Step 1: Choose classifier
            System.out.println("\nChoose a classifier:");
            System.out.println("1. Genetic Programming (GP)");
            System.out.println("2. Multi-Layer Perceptron (MLP)");
            System.out.println("3. J48 Decision Tree");
            System.out.println("4. Quit");
            System.out.print("Enter your choice (1-4): ");
            int choice;
            try {
                choice = scanner.nextInt();
            } catch (InputMismatchException e) {
                System.out.println("Invalid input. Please enter a number between 1 and 4.");
                scanner.nextLine(); // Consume the invalid input
                choice = 0; // Set a default value to trigger the reprompt
            }
            
            // Reprompt for invalid choice
            while (choice < 1 || choice > 4) {
                System.out.print("Invalid choice. Please enter a number between 1 and 4: ");
                choice = scanner.nextInt();
            }
            
            // Quit if option 4 is selected
            if (choice == 4) {
                System.out.println("Exiting program.");
                return;
            }
            
            // Step 2: Get seed value (if applicable)
            long seed = 0;
            if (choice == 1 || choice == 2 || choice == 3) { // GP, MLP, and J48 use seeds
                System.out.print("Enter seed value (or press Enter for random): ");
                scanner.nextLine(); // Consume the leftover newline
                String seedInput = scanner.nextLine().trim();
                if (seedInput.isEmpty()) {
                    seed = new Random().nextLong();
                    System.out.println("Using random seed: " + seed);
                } else {
                    seed = Long.parseLong(seedInput);
                }
            }
            Random random = new Random(seed);
            
            // Step 3: Get data file names (automatically prepend data/ directory)
            System.out.print("Enter training data filename (e.g., BTC_train.csv): ");
            String trainingFile = scanner.next();
            System.out.print("Enter test data filename (e.g., BTC_test.csv): ");
            String testFile = scanner.next();
            
            // Construct full paths
            String trainingPath = "data/" + trainingFile;
            String testPath = "data/" + testFile;
            
            // Check if files exist
            File trainingFileObj = new File(trainingPath);
            if (!trainingFileObj.exists()) {
                throw new Exception("Training file not found: " + trainingPath);
            }
            
            File testFileObj = new File(testPath);
            if (!testFileObj.exists()) {
                throw new Exception("Test file not found: " + testPath);
            }
            
            System.out.println("Loading training data from: " + trainingPath);
            
            // Load data
            DataSource source = new DataSource(trainingPath);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);
            
            System.out.println("Loaded " + data.numInstances() + " instances with " + data.numAttributes() + " attributes.");
            
            // Convert class attribute to nominal if it is numeric
            if (data.classAttribute().isNumeric()) {
                System.out.println("Converting class attribute from numeric to nominal...");
                NumericToNominal convert = new NumericToNominal();
                convert.setAttributeIndices("" + (data.classIndex() + 1)); // Weka uses 1-based indices
                convert.setInputFormat(data);
                data = Filter.useFilter(data, convert);
            }
            
            // Create results directory if it doesn't exist
            File resultsDir = new File("results");
            if (!resultsDir.exists()) {
                resultsDir.mkdir();
            }
            
            // Initialize and train selected classifier
            System.out.println("\nTraining selected classifier...");
            Evaluation eval = new Evaluation(data);
            String classifierName = "";
            String modelStructure = "";
            
            // Start timing
            long startTime = System.currentTimeMillis();
            
            try {
                switch (choice) {
                    case 1: // GP
                        System.out.println("Initializing GP Classifier with seed: " + seed);
                        GPClassifier gpClassifier = new GPClassifier();
                        gpClassifier.setSeed(seed);
                        gpClassifier.buildClassifier(data);
                        eval.crossValidateModel(gpClassifier, data, NUM_FOLDS, random);
                        classifierName = "GP Classifier";
                        modelStructure = gpClassifier.getModelStructure();
                        break;
                        
                    case 2: // MLP
                        System.out.println("Initializing MLP Classifier with seed: " + seed);
                        MLPWrapper mlpClassifier = new MLPWrapper();
                        mlpClassifier.setSeed(seed);
                        mlpClassifier.buildClassifier(data);
                        
                        // Manual cross-validation for MLP
                        eval = new Evaluation(data);
                        for (int i = 0; i < NUM_FOLDS; i++) {
                            Instances train = data.trainCV(NUM_FOLDS, i, random);
                            Instances test = data.testCV(NUM_FOLDS, i);
                            MLPWrapper foldClassifier = new MLPWrapper();
                            foldClassifier.buildClassifier(train);
                            foldClassifier.batchPredict(test);
                            
                            // Evaluate predictions
                            for (int j = 0; j < test.numInstances(); j++) {
                                double predicted = foldClassifier.classifyInstance(test.instance(j));
                                eval.evaluationForSingleInstance(new double[] {1.0 - predicted, predicted}, test.instance(j), true);
                            }
                        }
                        classifierName = "MLP Classifier";
                        modelStructure = mlpClassifier.getModelStructure();
                        break;
                        
                    case 3: // J48
                        System.out.println("Initializing J48 Classifier with seed: " + seed);
                        J48Classifier j48Classifier = new J48Classifier();
                        j48Classifier.setSeed(seed);
                        j48Classifier.buildClassifier(data);
                        eval.crossValidateModel(j48Classifier, data, NUM_FOLDS, random);
                        classifierName = "J48 Classifier";
                        modelStructure = j48Classifier.getModelStructure();
                        break;
                        
                    default:
                        throw new IllegalArgumentException("Invalid classifier choice");
                }
            } catch (Exception e) {
                System.err.println("Error training classifier: " + e.getMessage());
                e.printStackTrace();
                return;
            }
            
            // Calculate runtime
            long endTime = System.currentTimeMillis();
            double runtimeSeconds = (endTime - startTime) / 1000.0;
            
            // Print results
            System.out.println("\nResults for " + classifierName + ":");
            System.out.println("Runtime: " + String.format("%.2f", runtimeSeconds) + " seconds");
            System.out.println("Accuracy: " + eval.pctCorrect());
            System.out.println("F1 Score: " + eval.fMeasure(1));
            System.out.println("Buy Precision: " + eval.precision(1));
            System.out.println("Buy Recall: " + eval.recall(1));
            System.out.println("Model Structure: " + modelStructure);
            
            // Store results for statistical comparison if GP or MLP
            if (choice == 1 || choice == 2) {
                String resultKey = choice == 1 ? "GP" : "MLP";
                double[] predictions = new double[data.numInstances()];
                for (int i = 0; i < data.numInstances(); i++) {
                    predictions[i] = eval.predictions().get(i).predicted();
                }
                resultsMap.put(resultKey, predictions);
            }
            
            // Perform statistical comparison if both GP and MLP results are available
            if (resultsMap.size() == 2) {
                System.out.println("\nPerforming statistical comparison between GP and MLP:");
                double[] gpResults = resultsMap.get("GP");
                double[] mlpResults = resultsMap.get("MLP");
                
                StatisticalTest.WilcoxonResult result = StatisticalTest.wilcoxonSignedRankTest(
                    gpResults, mlpResults, ALPHA);
                
                System.out.println("Wilcoxon Signed-Rank Test Results:");
                System.out.println("Statistic: " + result.statistic);
                System.out.println("p-value: " + result.pValue);
                System.out.println("Significant difference: " + (result.isSignificant ? "Yes" : "No"));
                System.out.println("Interpretation: " + 
                    (result.isSignificant ? 
                        "There is a statistically significant difference between the classifiers (p < " + ALPHA + ")" :
                        "There is no statistically significant difference between the classifiers (p >= " + ALPHA + ")"));
            }
            
            // Save results to file
            String timestamp = java.time.LocalDateTime.now().format(java.time.format.DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
            String resultsPath = "results/classification_results_" + timestamp + ".txt";
            try (java.io.PrintWriter writer = new java.io.PrintWriter(new java.io.FileWriter(resultsPath))) {
                writer.println("Classification Results");
                writer.println("====================");
                writer.println("\nClassifier: " + classifierName);
                writer.println("Runtime: " + String.format("%.2f", runtimeSeconds) + " seconds");
                writer.println("Accuracy: " + eval.pctCorrect());
                writer.println("F1 Score: " + eval.fMeasure(1));
                writer.println("Buy Precision: " + eval.precision(1));
                writer.println("Buy Recall: " + eval.recall(1));
                writer.println("Model Structure: " + modelStructure);
                writer.println("\nConfusion Matrix:");
                writer.println(eval.toMatrixString());
            }
            System.out.println("\nResults have been saved to: " + resultsPath);
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void runMLPClassifier(Instances trainingData, Instances testData, long seed) throws Exception {
        MLPWrapper mlp = new MLPWrapper();
        mlp.setSeed(seed);
        mlp.buildClassifier(trainingData);
        mlp.batchPredict(testData); // Batch predict for all test instances
        System.out.println("\nMLP Classifier Results:");
        System.out.println("----------------------");
        System.out.println("Training Results:");
        evaluateClassifier(mlp, trainingData);
        System.out.println("\nTest Results:");
        evaluateClassifier(mlp, testData);
        // Store predictions for statistical comparison
        double[] predictions = new double[testData.numInstances()];
        for (int i = 0; i < testData.numInstances(); i++) {
            predictions[i] = mlp.classifyInstance(testData.instance(i));
        }
        resultsMap.put("MLP", predictions);
    }

    private static void evaluateClassifier(weka.classifiers.Classifier classifier, Instances data) throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(classifier, data);
        System.out.println("Accuracy: " + eval.pctCorrect());
        System.out.println("F1 Score: " + eval.fMeasure(1));
        System.out.println("Confusion Matrix:");
        System.out.println(eval.toMatrixString());
    }
}