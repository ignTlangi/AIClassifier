package classifiers;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import models.Individual;
import models.Population;
import java.util.List;
import java.util.ArrayList;
import java.util.Random;

public class GPClassifier extends AbstractClassifier {
    private Population population;
    private Individual bestIndividual;
    private Random random;
    private int populationSize = 100;
    private int generations = 50;
    private double mutationRate = 0.1;
    private double crossoverRate = 0.8;
    private long seed;

    public void setSeed(long seed) {
        this.seed = seed;
        this.random = new Random(seed);
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        if (data.classIndex() != data.numAttributes() - 1) {
            throw new Exception("Class attribute must be the last attribute");
        }

        // Convert Weka instances to DataPoints
        List<Individual.DataPoint> trainingData = new ArrayList<>();
        for (int i = 0; i < data.numInstances(); i++) {
            Instance instance = data.instance(i);
            double[] features = new double[data.numAttributes() - 1];
            for (int j = 0; j < data.numAttributes() - 1; j++) {
                features[j] = instance.value(j);
            }
            double label = instance.classValue();
            trainingData.add(new Individual.DataPoint(features, label));
        }

        // Initialize population
        population = new Population(populationSize, trainingData, random);
        
        // Evolve population
        for (int gen = 0; gen < generations; gen++) {
            population.evolve(mutationRate, crossoverRate);
        }

        // Store best individual
        bestIndividual = population.getBestIndividual();
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        if (bestIndividual == null) {
            throw new Exception("Classifier has not been built yet");
        }

        double[] features = new double[instance.numAttributes() - 1];
        for (int i = 0; i < instance.numAttributes() - 1; i++) {
            features[i] = instance.value(i);
        }

        Individual.DataPoint point = new Individual.DataPoint(features, 0);
        double prediction = bestIndividual.evaluate(point);
        return prediction > 0.5 ? 1.0 : 0.0;
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
        return bestIndividual != null ? bestIndividual.toString() : "Model not built";
    }
} 