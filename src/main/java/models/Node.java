package models;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

// TODO: Define the structure for a node in the GP program tree
public class Node {
    private String type; // e.g., "operator", "terminal_feature", "terminal_constant"
    private String value; // e.g., "+", "feature1", "0.5"
    private List<Node> children;
    private static Random random = new Random();

    // Constructor for a node
    public Node(String type, String value) {
        this.type = type;
        this.value = value;
        this.children = new ArrayList<>();
    }

    // Constructor for a placeholder node (used in Individual.java initially)
    public Node(String placeholderValue) {
        this.type = "placeholder";
        this.value = placeholderValue;
        this.children = new ArrayList<>();
    }

    // Adds a child to this node
    public void addChild(Node child) {
        this.children.add(child);
    }

    // Getters
    public String getType() {
        return type;
    }

    public String getValue() {
        return value;
    }

    public List<Node> getChildren() {
        return children;
    }

    // Evaluates this node (and its subtree) given a data point's features
    // This is a crucial part of how a GP tree makes a prediction or calculation
    public double evaluate(double[] features) {
        // TODO: Implement evaluation logic based on node type and value
        // Example:
        switch (type) {
            case "operator":
                // Apply operator to children's evaluations
                // e.g., if value is "+", return children.get(0).evaluate(features) + children.get(1).evaluate(features);
                // This needs to be robust for different operators and arities
                if (children.isEmpty()) return 0; // Or throw error
                if (value.equals("+") && children.size() >= 2) {
                    return children.get(0).evaluate(features) + children.get(1).evaluate(features);
                } else if (value.equals("-") && children.size() >= 2) {
                    return children.get(0).evaluate(features) - children.get(1).evaluate(features);
                } else if (value.equals("*") && children.size() >= 2) {
                    return children.get(0).evaluate(features) * children.get(1).evaluate(features);
                } else if (value.equals("/") && children.size() >= 2) {
                    double denominator = children.get(1).evaluate(features);
                    return denominator == 0 ? 1 : children.get(0).evaluate(features) / denominator; // Protected division
                }
                // Add more operators as needed (e.g., sin, cos, if-then-else)
                return 0; // Default for unhandled operators
            case "terminal_feature":
                // Return the value of the specified feature
                // e.g., if value is "feature0", return features[0];
                try {
                    int featureIndex = Integer.parseInt(value.replace("feature", ""));
                    if (featureIndex >= 0 && featureIndex < features.length) {
                        return features[featureIndex];
                    }
                } catch (NumberFormatException e) {
                    // Handle error if value is not a valid feature index
                }
                return 0; // Default or error value
            case "terminal_constant":
                // Return the constant value
                try {
                    return Double.parseDouble(value);
                } catch (NumberFormatException e) {
                    // Handle error
                }
                return 0; // Default or error value
            case "placeholder":
                // Placeholder nodes shouldn't typically be evaluated directly in a final tree
                return 0.0; // Or throw an error
            default:
                // Unknown node type
                return 0; // Or throw error
        }
    }

    // Method for deep copying a node and its children
    public Node copy() {
        Node newNode = new Node(this.type, this.value);
        for (Node child : this.children) {
            newNode.addChild(child.copy());
        }
        return newNode;
    }

    // Helper to represent the tree structure as a string (for debugging)
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(value);
        if (!children.isEmpty()) {
            sb.append("(");
            for (int i = 0; i < children.size(); i++) {
                sb.append(children.get(i).toString());
                if (i < children.size() - 1) {
                    sb.append(", ");
                }
            }
            sb.append(")");
        }
        return sb.toString();
    }
}