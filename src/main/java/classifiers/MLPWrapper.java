package classifiers;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import java.io.File;
import java.io.FileWriter;
import java.io.BufferedReader;
import java.io.InputStreamReader;

public class MLPWrapper extends AbstractClassifier {
    private String modelPath;
    private String tempDataPath;
    private long seed;

    public void setSeed(long seed) {
        this.seed = seed;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        if (data.classIndex() != data.numAttributes() - 1) {
            throw new Exception("Class attribute must be the last attribute");
        }

        // Create temporary directory if it doesn't exist
        File tempDir = new File("temp");
        if (!tempDir.exists()) {
            tempDir.mkdir();
        }

        // Save data to temporary CSV file
        tempDataPath = "temp/temp_data.csv";
        try (FileWriter writer = new FileWriter(tempDataPath)) {
            // Write header
            for (int i = 0; i < data.numAttributes() - 1; i++) {
                writer.write("feature" + i + ",");
            }
            writer.write("class\n");

            // Write data
            for (int i = 0; i < data.numInstances(); i++) {
                Instance instance = data.instance(i);
                for (int j = 0; j < data.numAttributes() - 1; j++) {
                    writer.write(instance.value(j) + ",");
                }
                writer.write(instance.classValue() + "\n");
            }
        }

        // Set model path
        modelPath = "temp/mlp_model.pkl";

        // Run Python script
        ProcessBuilder pb = new ProcessBuilder("python", "src/main/python/mlp_classifier.py", tempDataPath, modelPath);
        Process p = pb.start();

        // Read output
        BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()));
        String line;
        while ((line = reader.readLine()) != null) {
            System.out.println(line);
        }

        // Wait for process to complete
        int exitCode = p.waitFor();
        if (exitCode != 0) {
            throw new Exception("Python script failed with exit code " + exitCode);
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        if (modelPath == null) {
            throw new Exception("Classifier has not been built yet");
        }

        // Save instance to temporary file
        String tempInstancePath = "temp/temp_instance.csv";
        try (FileWriter writer = new FileWriter(tempInstancePath)) {
            // Write header
            for (int i = 0; i < instance.numAttributes() - 1; i++) {
                writer.write("feature" + i + ",");
            }
            writer.write("class\n");

            // Write instance
            for (int i = 0; i < instance.numAttributes() - 1; i++) {
                writer.write(instance.value(i) + ",");
            }
            writer.write("0\n"); // Dummy class value
        }

        // Run Python script for prediction
        ProcessBuilder pb = new ProcessBuilder("python", "src/main/python/mlp_classifier.py", tempInstancePath, modelPath, "--predict");
        Process p = pb.start();

        // Read prediction
        BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()));
        String prediction = reader.readLine();
        if (prediction == null) {
            throw new Exception("No prediction received from Python script");
        }

        return Double.parseDouble(prediction);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double pred = classifyInstance(instance);
        return new double[] {1.0 - pred, pred};
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = new Capabilities(this);
        result.disableAll();
        // Enable numeric attributes
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        // Enable binary class
        result.enable(Capability.BINARY_CLASS);
        return result;
    }

    public String getModelStructure() {
        return "MLP Classifier (Python)";
    }
} 