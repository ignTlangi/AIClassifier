package main.java.models;

import java.util.List;
import java.util.Random;
import java.util.ArrayList;
import java.io.Serializable;

// TODO: Define the structure for an individual in the GP population
public class Individual implements Serializable {
    private static final long serialVersionUID = 1L;
    // Represents the program tree for this individual
    private Node root;
    private double fitness;
    private static final int MAX_DEPTH = 5;
    private static final String[] OPERATORS = {"+", "-", "*", "/"};
    private static final Random random = new Random();

    // Constructor for creating an individual
    public Individual() {
        this.root = null;
        this.fitness = 0.0;
    }

    public Individual(Node root) {
        this.root = root;
        this.fitness = 0.0;
    }

    public static Individual generateRandom(int numFeatures, Random random) {
        Node root = generateRandomNode(0, numFeatures, random);
        return new Individual(root);
    }

    private static Node generateRandomNode(int depth, int numFeatures, Random random) {
        if (depth >= MAX_DEPTH || (depth > 0 && random.nextDouble() < 0.3)) {
            // Generate terminal node (feature or constant)
            if (random.nextDouble() < 0.5) {
                // Feature
                return new Node(String.valueOf(random.nextInt(numFeatures)));
            } else {
                // Constant
                return new Node(String.valueOf(random.nextDouble() * 10));
            }
        } else {
            // Generate operator node
            String operator = OPERATORS[random.nextInt(OPERATORS.length)];
            Node node = new Node(operator);
            node.left = generateRandomNode(depth + 1, numFeatures, random);
            node.right = generateRandomNode(depth + 1, numFeatures, random);
            return node;
        }
    }

    // Evaluates the fitness of the individual
    public double evaluate(DataPoint point) {
        if (root == null) {
            return 0.5; // Default prediction if no tree exists
        }
        try {
            return evaluateNode(root, point);
        } catch (Exception e) {
            System.err.println("Error evaluating node: " + e.getMessage());
            return 0.5; // Default prediction on error
        }
    }

    private double evaluateNode(Node node, DataPoint point) {
        if (node == null) {
            return 0; // or some default value
        }
        if (node.isTerminal()) {
            try {
                // Try to parse as feature index
                int featureIndex = Integer.parseInt(node.value);
                if (featureIndex >= 0 && featureIndex < point.features.length) {
                    return point.features[featureIndex];
                } else {
                    System.err.println("Feature index out of bounds: " + featureIndex + 
                                       " (max: " + (point.features.length-1) + ")");
                    return 0.0;
                }
            } catch (NumberFormatException e) {
                try {
                    // Try to parse as constant
                    return Double.parseDouble(node.value);
                } catch (NumberFormatException e2) {
                    System.err.println("Invalid node value: " + node.value);
                    return 0.0;
                }
            }
        }
        
        // Process operators
        double leftValue = evaluateNode(node.left, point);
        double rightValue = evaluateNode(node.right, point);
        
        // Check if the node value is actually an operator
        if (node.value.equals("+")) {
            return leftValue + rightValue;
        } else if (node.value.equals("-")) {
            return leftValue - rightValue;
        } else if (node.value.equals("*")) {
            return leftValue * rightValue;
        } else if (node.value.equals("/")) {
            if (Math.abs(rightValue) < 1e-10) {
                // Avoid division by zero
                return 1.0; 
            }
            return leftValue / rightValue;
        } else {
            // Try to parse as a number if not a recognized operator
            try {
                return Double.parseDouble(node.value);
            } catch (NumberFormatException e) {
                System.err.println("Unknown operator: " + node.value);
                return 0;
            }
        }
    }

    // Getter for fitness
    public double getFitness() {
        return fitness;
    }

    // Setter for fitness (if needed, e.g., after evaluation)
    public void setFitness(double fitness) {
        this.fitness = fitness;
    }

    // Getter for the root node of the program tree
    public Node getRoot() {
        return root;
    }

    // Setter for the root node (e.g., after crossover or mutation)
    public void setRoot(Node root) {
        this.root = root;
    }

    // Method for deep copying an individual (important for selection/reproduction)
    public Individual deepCopy() {
        return new Individual(root != null ? root.deepCopy() : null);
    }

    public Node getRandomNode(Random random) {
        List<Node> nodes = new ArrayList<>();
        collectNodes(root, nodes);
        if (nodes.isEmpty()) {
            return null;
        }
        return nodes.get(random.nextInt(nodes.size()));
    }

    private void collectNodes(Node node, List<Node> nodes) {
        if (node == null) return;
        nodes.add(node);
        collectNodes(node.left, nodes);
        collectNodes(node.right, nodes);
    }

    @Override
    public String toString() {
        return root != null ? root.toString() : "Empty";
    }

    // DataPoint class for representing training and testing data
    public static class DataPoint implements Serializable {
        private static final long serialVersionUID = 1L;
        public final double[] features;
        public final double label; // e.g., 0 for sell, 1 for buy

        public DataPoint(double[] features, double label) {
            this.features = features;
            this.label = label;
        }
    }

    public static class Node implements Serializable {
        private static final long serialVersionUID = 1L;
        public String value;
        public Node left;
        public Node right;

        public Node(String value) {
            this.value = value;
            this.left = null;
            this.right = null;
        }

        public boolean isTerminal() {
            return left == null && right == null;
        }

        public Node deepCopy() {
            Node copy = new Node(value);
            if (left != null) copy.left = left.deepCopy();
            if (right != null) copy.right = right.deepCopy();
            return copy;
        }

        @Override
        public String toString() {
            if (isTerminal()) {
                return value;
            }
            
            String leftStr = left != null ? left.toString() : "null";
            String rightStr = right != null ? right.toString() : "null";
            return "(" + leftStr + " " + value + " " + rightStr + ")";
        }
    }
}