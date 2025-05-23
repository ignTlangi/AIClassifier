package classifiers;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;

public class J48Classifier extends AbstractClassifier {
    private J48 j48;
    private boolean isTrained;
    private long seed;

    public J48Classifier() {
        j48 = new J48();
        isTrained = false;
        // Set improved parameters
        j48.setConfidenceFactor(0.1f);  // Lower confidence factor to prevent over-pruning
        j48.setMinNumObj(10);           // Increase minimum instances per leaf
        j48.setUnpruned(false);         // Keep pruning but less aggressive
        j48.setUseLaplace(true);        // Use Laplace smoothing for better probability estimates
    }

    public void setSeed(long seed) {
        this.seed = seed;
        // Only set the seed if reduced error pruning is enabled
        if (j48.getReducedErrorPruning()) {
            this.j48.setSeed((int) seed);
        }
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        if (data.classIndex() != data.numAttributes() - 1) {
            throw new Exception("Class attribute must be the last attribute");
        }
        j48.buildClassifier(data);
        isTrained = true;
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        if (!isTrained) {
            throw new IllegalStateException("Classifier must be trained before classification");
        }
        return j48.classifyInstance(instance);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        if (!isTrained) {
            throw new IllegalStateException("Classifier must be trained before classification");
        }
        return j48.distributionForInstance(instance);
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = new Capabilities(this);
        result.disableAll();
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.BINARY_CLASS);
        result.enable(Capability.MISSING_VALUES);
        return result;
    }

    public String getModelStructure() {
        return isTrained ? j48.toString() : "Model not trained";
    }

    public boolean isTrained() {
        return isTrained;
    }
}