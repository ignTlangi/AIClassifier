package classifiers;

import weka.core.Instance;
import java.util.Random;
import java.util.ArrayList;
import java.util.List;

public class GPNode {
    private static final String[] OPERATORS = {"+", "-", "*", "/"};
    private static final int MAX_DEPTH = 5;
    
    private String value;
    private List<GPNode> children;
    private GPNode parent;
    private double fitness;
    
    public GPNode(String value) {
        this.value = value;
        this.children = new ArrayList<>();
        this.fitness = 0.0;
    }
    
    public static GPNode generateRandomTree(int numAttributes, Random random) {
        return generateRandomNode(0, numAttributes, random);
    }
    
    private static GPNode generateRandomNode(int depth, int numAttributes, Random random) {
        if (depth >= MAX_DEPTH || (depth > 0 && random.nextDouble() < 0.3)) {
            // Generate terminal node (attribute or constant)
            if (random.nextDouble() < 0.5) {
                // Attribute
                int attrIndex = random.nextInt(numAttributes);
                return new GPNode("x" + attrIndex);
            } else {
                // Constant
                double constant = random.nextDouble() * 2 - 1; // Random value between -1 and 1
                return new GPNode(String.valueOf(constant));
            }
        } else {
            // Generate operator node
            String operator = OPERATORS[random.nextInt(OPERATORS.length)];
            GPNode node = new GPNode(operator);
            
            // Add children
            node.addChild(generateRandomNode(depth + 1, numAttributes, random));
            node.addChild(generateRandomNode(depth + 1, numAttributes, random));
            
            return node;
        }
    }
    
    public void addChild(GPNode child) {
        children.add(child);
        child.setParent(this);
    }
    
    public void setParent(GPNode parent) {
        this.parent = parent;
    }
    
    public GPNode getParent() {
        return parent;
    }
    
    public void setFitness(double fitness) {
        this.fitness = fitness;
    }
    
    public double getFitness() {
        return fitness;
    }
    
    public double evaluate(Instance instance) {
        if (children.isEmpty()) {
            // Terminal node
            if (value.startsWith("x")) {
                // Attribute
                int attrIndex = Integer.parseInt(value.substring(1));
                return instance.value(attrIndex);
            } else {
                // Constant
                return Double.parseDouble(value);
            }
        } else {
            // Operator node
            double left = children.get(0).evaluate(instance);
            double right = children.get(1).evaluate(instance);
            
            switch (value) {
                case "+": return left + right;
                case "-": return left - right;
                case "*": return left * right;
                case "/": return right != 0 ? left / right : 0;
                default: return 0;
            }
        }
    }
    
    public GPNode deepCopy() {
        GPNode copy = new GPNode(value);
        for (GPNode child : children) {
            copy.addChild(child.deepCopy());
        }
        return copy;
    }
    
    public GPNode getRandomNode(Random random) {
        List<GPNode> allNodes = new ArrayList<>();
        collectNodes(this, allNodes);
        return allNodes.get(random.nextInt(allNodes.size()));
    }
    
    private void collectNodes(GPNode node, List<GPNode> nodes) {
        nodes.add(node);
        for (GPNode child : node.children) {
            collectNodes(child, nodes);
        }
    }
    
    public void replaceWith(GPNode newNode) {
        if (parent != null) {
            int index = parent.children.indexOf(this);
            parent.children.set(index, newNode);
            newNode.setParent(parent);
        }
    }
    
    @Override
    public String toString() {
        if (children.isEmpty()) {
            return value;
        } else {
            return "(" + children.get(0).toString() + " " + value + " " + children.get(1).toString() + ")";
        }
    }
} 