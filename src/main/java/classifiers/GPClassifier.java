package classifiers;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import main.java.models.Individual;
import main.java.models.Population;
import java.util.List;
import java.util.ArrayList;
import java.util.Random;

public class GPClassifier extends AbstractClassifier {
    private Population population;
    private Individual bestIndividual;
    private Random random;
    private int populationSize = 50;
    private int generations = 30;
    private double mutationRate = 0.2;
    private double crossoverRate = 0.8;
    private long seed;
    private int noImprovementLimit = 8;

    public void setSeed(long seed) {
        this.seed = seed;
        this.random = new Random(seed);
    }

    /**
     * Set the early stopping threshold (number of generations with no improvement before stopping)
     */
    public void setNoImprovementLimit(int limit) {
        this.noImprovementLimit = limit;
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
        double bestFitness = Double.NEGATIVE_INFINITY;
        int noImprovementCount = 0;
        int actualGenerations = 0;
        System.out.println("[GP] Early stopping threshold: " + noImprovementLimit + " generations with no improvement.");
        for (int gen = 0; gen < generations; gen++) {
            population.evolve(mutationRate, crossoverRate);
            double currentBestFitness = population.getBestIndividual().getFitness();
            System.out.println("Generation " + gen + ", Best Fitness: " + currentBestFitness);
            actualGenerations++;
            if (currentBestFitness > bestFitness) {
                bestFitness = currentBestFitness;
                noImprovementCount = 0;
            } else {
                noImprovementCount++;
            }
            if (noImprovementCount >= noImprovementLimit) { // Configurable early stopping
                System.out.println("[GP] Stopping early at generation " + gen + " due to no improvement for " + noImprovementLimit + " generations.");
                break;
            }
        }
        System.out.println("[GP] Training completed after " + actualGenerations + " generations (max allowed: " + generations + ").");
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