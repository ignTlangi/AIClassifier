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
import java.nio.file.Paths;

public class MLPWrapper extends AbstractClassifier {
    private String modelPath;
    private String tempDataPath;
    private long seed;
    private String pythonScriptPath;
    private boolean isTrained = false;
    private double[] cachedPredictions = null;
    private int predictionIndex = 0;

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

        // Get Python executable - try python or python3
        String pythonCmd = "python";
        try {
            ProcessBuilder pbTest = new ProcessBuilder(pythonCmd, "--version");
            Process pTest = pbTest.start();
            if (pTest.waitFor() != 0) {
                pythonCmd = "python3";
            }
        } catch (Exception e) {
            System.out.println("Trying fallback to python3");
            pythonCmd = "python3";
        }

        // Find the Python script using multiple possible locations
        File scriptFile = findPythonScript();
        if (scriptFile == null || !scriptFile.exists()) {
            throw new Exception("Could not find the Python script at any of the expected locations");
        }
        
        pythonScriptPath = scriptFile.getAbsolutePath();
        System.out.println("Using Python script at: " + pythonScriptPath);

        // Run Python script with seed
        ProcessBuilder pb = new ProcessBuilder(pythonCmd, pythonScriptPath, tempDataPath, modelPath, String.valueOf(seed));
        pb.redirectErrorStream(true); // Merge error stream with input stream
        System.out.println("Running command: " + String.join(" ", pb.command()));
        
        Process p = pb.start();

        // Read output
        BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()));
        String line;
        while ((line = reader.readLine()) != null) {
            System.out.println("Python: " + line);
        }

        // Wait for process to complete
        int exitCode = p.waitFor();
        if (exitCode != 0) {
            throw new Exception("Python script failed with exit code " + exitCode);
        }

        isTrained = true;
        // Clear prediction cache on new training
        cachedPredictions = null;
        predictionIndex = 0;
    }
    
    private File findPythonScript() {
        // Try multiple possible locations for the script
        String[] possiblePaths = {
            "src/main/python/mlp_classifier.py",
            "python/mlp_classifier.py",
            "src/python/mlp_classifier.py",
            Paths.get(System.getProperty("user.dir"), "src", "main", "python", "mlp_classifier.py").toString()
        };
        
        for (String path : possiblePaths) {
            File file = new File(path);
            if (file.exists()) {
                return file;
            }
        }
        
        // If we can't find it, try to get the current working directory and look there
        String currentDir = System.getProperty("user.dir");
        File dir = new File(currentDir);
        File[] files = dir.listFiles((d, name) -> name.equals("mlp_classifier.py"));
        if (files != null && files.length > 0) {
            return files[0];
        }
        
        return null;
    }

    public void batchPredict(Instances instances) throws Exception {
        // Save all instances to a temp CSV
        String tempBatchPath = "temp/temp_batch.csv";
        try (java.io.PrintWriter writer = new java.io.PrintWriter(new java.io.FileWriter(tempBatchPath))) {
            // Write header
            for (int i = 0; i < instances.numAttributes() - 1; i++) {
                writer.write("feature" + i + ",");
            }
            writer.write("class\n");
            // Write data
            for (int i = 0; i < instances.numInstances(); i++) {
                Instance instance = instances.instance(i);
                for (int j = 0; j < instances.numAttributes() - 1; j++) {
                    writer.write(instance.value(j) + ",");
                }
                writer.write("0\n"); // Dummy class value
            }
        }
        // Run prediction command
        String pythonCmd = "python";
        try {
            ProcessBuilder pbTest = new ProcessBuilder(pythonCmd, "--version");
            Process pTest = pbTest.start();
            if (pTest.waitFor() != 0) {
                pythonCmd = "python3";
            }
        } catch (Exception e) {
            pythonCmd = "python3";
        }
        if (pythonScriptPath == null || pythonScriptPath.isEmpty()) {
            File scriptFile = findPythonScript();
            if (scriptFile == null || !scriptFile.exists()) {
                throw new Exception("Could not find the Python script for prediction");
            }
            pythonScriptPath = scriptFile.getAbsolutePath();
        }
        ProcessBuilder pb = new ProcessBuilder(pythonCmd, pythonScriptPath, tempBatchPath, modelPath, String.valueOf(seed), "--predict");
        pb.redirectErrorStream(true);
        Process p = pb.start();
        java.io.BufferedReader reader = new java.io.BufferedReader(new java.io.InputStreamReader(p.getInputStream()));
        String line;
        java.util.List<Double> preds = new java.util.ArrayList<>();
        while ((line = reader.readLine()) != null) {
            if (line.trim().matches("\\d+(\\.\\d+)?")) {
                preds.add(Double.parseDouble(line.trim()));
            }
        }
        p.waitFor();
        cachedPredictions = preds.stream().mapToDouble(Double::doubleValue).toArray();
        predictionIndex = 0;
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        if (!isTrained) {
            throw new IllegalStateException("Classifier must be trained before classification");
        }
        if (cachedPredictions == null || predictionIndex >= cachedPredictions.length) {
            // Create a single-instance dataset for prediction
            Instances singleInstance = new Instances(instance.dataset(), 0);
            singleInstance.add(instance);
            batchPredict(singleInstance);
        }
        return cachedPredictions[predictionIndex++];
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