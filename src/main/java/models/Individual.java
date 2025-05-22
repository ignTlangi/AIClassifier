package models;

import java.util.List;
import java.util.Random;
import java.util.ArrayList;

// TODO: Define the structure for an individual in the GP population
public class Individual {
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
        return evaluateNode(root, point);
    }

    private double evaluateNode(Node node, DataPoint point) {
        if (node.isTerminal()) {
            try {
                int featureIndex = Integer.parseInt(node.value);
                return point.features[featureIndex];
            } catch (NumberFormatException e) {
                return Double.parseDouble(node.value);
            }
        }

        double leftValue = evaluateNode(node.left, point);
        double rightValue = evaluateNode(node.right, point);

        switch (node.value) {
            case "+": return leftValue + rightValue;
            case "-": return leftValue - rightValue;
            case "*": return leftValue * rightValue;
            case "/": return rightValue != 0 ? leftValue / rightValue : 0;
            default: return 0;
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
        return new Individual(root.deepCopy());
    }

    public Node getRandomNode(Random random) {
        List<Node> nodes = new ArrayList<>();
        collectNodes(root, nodes);
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
    public static class DataPoint {
        // Fields for stock data features and classification label
        public final double[] features;
        public final double label; // e.g., 0 for sell, 1 for buy

        public DataPoint(double[] features, double label) {
            this.features = features;
            this.label = label;
        }
    }

    public static class Node {
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
            return "(" + left.toString() + " " + value + " " + right.toString() + ")";
        }
    }
}