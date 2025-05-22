package main.java.models;

import java.util.List;
import java.util.ArrayList;
import java.util.Random;
import java.io.Serializable;

public class Population implements Serializable {
    private static final long serialVersionUID = 1L;
    private List<Individual> individuals;
    private List<Individual.DataPoint> trainingData;
    private Random random;
    private Individual bestIndividual;

    public Population(int size, List<Individual.DataPoint> trainingData, Random random) {
        this.trainingData = trainingData;
        this.random = random;
        this.individuals = new ArrayList<>();
        
        // Initialize population with random individuals
        for (int i = 0; i < size; i++) {
            individuals.add(Individual.generateRandom(trainingData.get(0).features.length, random));
        }
        
        // Evaluate initial population
        evaluatePopulation();
    }

    private void evaluatePopulation() {
        bestIndividual = null;
        double bestFitness = Double.NEGATIVE_INFINITY;
        
        for (Individual individual : individuals) {
            double fitness = evaluateFitness(individual);
            individual.setFitness(fitness);
            
            if (fitness > bestFitness) {
                bestFitness = fitness;
                bestIndividual = individual;
            }
        }
    }

    private double evaluateFitness(Individual individual) {
        int correctPredictions = 0;
        
        for (Individual.DataPoint point : trainingData) {
            double prediction = individual.evaluate(point);
            double classifiedAs = prediction > 0.5 ? 1.0 : 0.0;
            
            if (Math.abs(classifiedAs - point.label) < 0.01) {
                correctPredictions++;
            }
        }
        
        return (double) correctPredictions / trainingData.size();
    }

    public void evolve(double mutationRate, double crossoverRate) {
        List<Individual> newPopulation = new ArrayList<>();
        
        // Elitism: keep the best individual
        newPopulation.add(bestIndividual.deepCopy());
        
        // Generate rest of the population
        while (newPopulation.size() < individuals.size()) {
            if (random.nextDouble() < crossoverRate) {
                // Crossover
                Individual parent1 = tournamentSelection();
                Individual parent2 = tournamentSelection();
                Individual[] children = crossover(parent1, parent2);
                newPopulation.add(children[0]);
                if (newPopulation.size() < individuals.size()) {
                    newPopulation.add(children[1]);
                }
            } else {
                // Mutation
                Individual individual = tournamentSelection();
                Individual mutated = mutate(individual, mutationRate);
                newPopulation.add(mutated);
            }
        }
        
        individuals = newPopulation;
        evaluatePopulation();
    }

    private Individual tournamentSelection() {
        int tournamentSize = 3;
        Individual best = individuals.get(random.nextInt(individuals.size()));
        
        for (int i = 1; i < tournamentSize; i++) {
            Individual candidate = individuals.get(random.nextInt(individuals.size()));
            if (candidate.getFitness() > best.getFitness()) {
                best = candidate;
            }
        }
        
        return best;
    }

    private Individual[] crossover(Individual parent1, Individual parent2) {
        Individual child1 = parent1.deepCopy();
        Individual child2 = parent2.deepCopy();
        
        // Select random crossover points
        Individual.Node point1 = child1.getRandomNode(random);
        Individual.Node point2 = child2.getRandomNode(random);
        
        // Perform crossover
        Individual.Node temp = point1.left;
        point1.left = point2.left;
        point2.left = temp;
        
        return new Individual[]{child1, child2};
    }

    private Individual mutate(Individual individual, double mutationRate) {
        Individual mutated = individual.deepCopy();
        Individual.Node mutationPoint = mutated.getRandomNode(random);
        
        if (random.nextDouble() < mutationRate) {
            // Replace with a new random subtree
            Individual.Node newNode = Individual.generateRandom(trainingData.get(0).features.length, random).getRandomNode(random);
            mutationPoint.left = newNode.left;
            mutationPoint.right = newNode.right;
            mutationPoint.value = newNode.value;
        }
        
        return mutated;
    }

    public Individual getBestIndividual() {
        return bestIndividual;
    }
} 