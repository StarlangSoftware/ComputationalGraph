package ComputationalGraph;

import Math.Tensor;
import java.util.*;

public class ComputationalGraph {
    private final HashMap<ComputationalNode, List<ComputationalNode>> nodeMap = new HashMap<>();
    private final HashMap<ComputationalNode, List<ComputationalNode>> reverseNodeMap = new HashMap<>();

    public ComputationalNode addEdge(ComputationalNode first, Object second, boolean isBiased) {
        ComputationalNode newNode;
        if (second instanceof Function) {
            newNode = new ComputationalNode(false, isBiased, null, (Function) second, null);
        } else if (second instanceof ComputationalNode) {
            newNode = new ComputationalNode(false, isBiased, ((ComputationalNode) second).getOperator(), null, null);
        } else {
            throw new IllegalArgumentException("Invalid type for 'second'. Must be a ComputationalNode or FunctionType.");
        }
        nodeMap.computeIfAbsent(first, k -> new ArrayList<>()).add(newNode);
        reverseNodeMap.computeIfAbsent(newNode, k -> new ArrayList<>()).add(first);

        if (second instanceof ComputationalNode) {
            nodeMap.computeIfAbsent((ComputationalNode) second, k -> new ArrayList<>()).add(newNode);
            reverseNodeMap.computeIfAbsent(newNode, k -> new ArrayList<>()).add((ComputationalNode) second);
        }
        return newNode;
    }


    /**
     * Recursive helper function to perform depth-first search for topological sorting.
     * @param node The current node being processed.
     * @param visited A set of visited nodes.
     * @return A list representing the partial topological order.
     */
    private LinkedList<ComputationalNode> sortRecursive(ComputationalNode node, Set<ComputationalNode> visited) {
        LinkedList<ComputationalNode> queue = new LinkedList<>();
        visited.add(node);
        if (nodeMap.containsKey(node)) {
            for (ComputationalNode child : nodeMap.get(node)) {
                if (!visited.contains(child)) {
                    queue.addAll(sortRecursive(child, visited));
                }
            }
        }
        queue.offer(node);
        return queue;
    }

    /**
     * Performs topological sorting on the computational graph.
     * @return A list representing the topological order of the nodes.
     */
    public ArrayList<ComputationalNode> topologicalSort() {
        LinkedList<ComputationalNode> sortedList = new LinkedList<>();
        HashSet<ComputationalNode> visited = new HashSet<>();
        for (ComputationalNode node : nodeMap.keySet()) {
            if (!visited.contains(node)) {
                LinkedList<ComputationalNode> queue = sortRecursive(node, visited);
                while (!queue.isEmpty()) {
                    sortedList.offerLast(queue.pollFirst());
                }
            }
        }
        return new ArrayList<>(sortedList);
    }

    /**
     * Recursive helper function to clear the values and gradients of nodes.
     */
    private void clearRecursive(Set<ComputationalNode> visited, ComputationalNode node) {
        visited.add(node);
        if (!node.isLearnable()) {
            node.setValue(null);
        }
        node.setBackward(null);
        if (nodeMap.containsKey(node)) {
            for (ComputationalNode child : nodeMap.get(node)) {
                if (!visited.contains(child)) {
                    clearRecursive(visited, child);
                }
            }
        }
    }

    /**
     * Clears the values and gradients of all nodes in the graph.
     */
    public void clear() {
        HashSet<ComputationalNode> visited = new HashSet<>();
        for (ComputationalNode node : nodeMap.keySet()) {
            if (!visited.contains(node)) {
                clearRecursive(visited, node);
            }
        }
    }

    /**
     * Recursive helper function to update the values of learnable nodes.
     */
    private void updateRecursive(Set<ComputationalNode> visited, ComputationalNode node) {
        visited.add(node);
        if (node.isLearnable()) {
            node.updateValue();
        }
        if (nodeMap.containsKey(node)) {
            for (ComputationalNode child : nodeMap.get(node)) {
                if (!visited.contains(child)) {
                    updateRecursive(visited, child);
                }
            }
        }
    }

    /**
     * Updates the values of all learnable nodes in the graph.
     */
    public void updateValues() {
        HashSet<ComputationalNode> visited = new HashSet<>();
        for (ComputationalNode node : nodeMap.keySet()) {
            if (!visited.contains(node)) {
                updateRecursive(visited, node);
            }
        }
    }

    /**
     * Calculates the derivative of the child node with respect to the parent node.
     * @param node Parent node.
     * @param child Child node.
     * @return The gradient tensor.
     */
    public Tensor calculateDerivative(ComputationalNode node, ComputationalNode child) {
        List<ComputationalNode> reverseChildren = reverseNodeMap.get(child);
        if (reverseChildren == null || reverseChildren.isEmpty()) {
            return null;
        }
        ComputationalNode left = reverseChildren.get(0);
        if (reverseChildren.size() == 1) {
            Function function = child.getFunction();
            Tensor backward = child.getBackward();
            Tensor derivative = function.derivative(child.getValue());
            if (backward != null && derivative != null) {
                return backward.multiply(derivative); // Optimized element-wise multiplication
            }
            return null;
        } else {
            ComputationalNode right = reverseChildren.get(1);
            if (child.getOperator() != null) {
                switch (child.getOperator()) {
                    case "*":
                        if (left == node) {
                            if (!child.isBiased()) {
                                Tensor backward = child.getBackward();
                                Tensor rightValue = right.getValue();
                                if (backward != null && rightValue != null) {
                                    return backward.dot(rightValue.transpose(null));
                                }
                                return null;
                            }
                            Tensor backward = child.getBackward();
                            Tensor partial = backward.partial(new int[]{0, 0}, new int[]{backward.getShape()[0], backward.getShape()[1] - 1});
                            Tensor rightValue = right.getValue();
                            if (partial != null && rightValue != null) {
                                return partial.dot(rightValue.transpose(null));
                            }
                            return null;
                        }
                        Tensor leftValue = left.getValue();
                        Tensor backward = child.getBackward();
                        if (leftValue != null && backward != null) {
                            return leftValue.transpose(null).dot(backward);
                        }
                        return null;
                    case "+":
                        return child.getBackward();
                    case "-":
                        if (left == node) {
                            return child.getBackward();
                        } else {
                            Tensor result = child.getBackward();
                            if (result != null) {
                                int[] shape = result.getShape();
                                for (int i = 0; i < shape[0]; i++) {
                                    for (int j = 0; j < shape[1]; j++) {
                                        result.set(new int[]{i, j}, -result.get(new int[]{i, j}));
                                    }
                                }
                                return result;
                            }
                            return null;
                        }
                }
            }
        }
        return null;
    }

    /**
     * Computes the difference between the predicted and actual values (R - Y).
     * @param output The output node of the computational graph.
     * @param learningRate The learning rate for gradient descent.
     * @param classLabelIndex A list of true class labels (index of the correct class for each sample).
     */
    public void calculateRMinusY(ComputationalNode output, double learningRate, List<Integer> classLabelIndex) {
        Tensor outputValue = output.getValue();
        if (outputValue == null) return;
        int rows = outputValue.getShape()[0];
        int cols = outputValue.getShape()[1];
        List<List<Double>> initialBackwardData = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            List<Double> row = new ArrayList<>();
            for (int j = 0; j < cols; j++) {
                row.add(0.0);
            }
            initialBackwardData.add(row);
        }
        Tensor backward = new Tensor(initialBackwardData, new int[]{rows, cols});
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (classLabelIndex.get(i) == j) {
                    backward.set(new int[]{i, j}, (1 - outputValue.get(new int[]{i, j})) * learningRate);
                } else {
                    backward.set(new int[]{i, j}, (-outputValue.get(new int[]{i, j})) * learningRate);
                }
            }
        }
        output.setBackward(backward);
    }

    /**
     * Performs backpropagation on the computational graph.
     * @param learningRate The learning rate for gradient descent.
     * :param classLabelIndex: The true class labels (as a list of integers).
     */
    public void backpropagation(double learningRate, List<Integer> classLabelIndex) {
        List<ComputationalNode> sortedNodes = topologicalSort();
        if (sortedNodes.isEmpty()) return;
        ComputationalNode outputNode = sortedNodes.remove(0);
        calculateRMinusY(outputNode, learningRate, classLabelIndex);
        if (!sortedNodes.isEmpty()) {
            sortedNodes.remove(0).setBackward(outputNode.getBackward());
        }
        while (!sortedNodes.isEmpty()) {
            ComputationalNode node = sortedNodes.remove(0);
            List<ComputationalNode> children = nodeMap.get(node);
            if (children != null) {
                for (ComputationalNode child : children) {
                    Tensor derivative = calculateDerivative(node, child);
                    if (derivative != null) {
                        if (node.getBackward() == null) {
                            node.setBackward(derivative);
                        } else {
                            Tensor currentBackward = node.getBackward();
                            int[] shape = currentBackward.getShape();
                            for (int i = 0; i < shape[0]; i++) {
                                for (int j = 0; j < shape[1]; j++) {
                                    currentBackward.set(new int[]{i, j}, currentBackward.get(new int[]{i, j}) + derivative.get(new int[]{i, j}));
                                }
                            }
                        }
                    }
                }
            }
        }

        updateValues();
        clear();
    }

    /**
     * Add a bias term to the node's value by appending a column of ones.
     */
    public void getBiased(ComputationalNode first) {
        Tensor firstValue = first.getValue();
        if (firstValue == null) return;
        int rows = firstValue.getShape()[0];
        int originalCols = firstValue.getShape()[1];
        int newCols = originalCols + 1;
        List<List<Double>> initialBiasedValueData = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            List<Double> row = new ArrayList<>();
            for (int j = 0; j < newCols; j++) {
                row.add(0.0);
            }
            initialBiasedValueData.add(row);
        }
        Tensor biasedValue = new Tensor(initialBiasedValueData, new int[]{rows, newCols});
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < originalCols; j++) {
                biasedValue.set(new int[]{i, j}, firstValue.get(new int[]{i, j}));
            }
            biasedValue.set(new int[]{i, originalCols}, 1.0);
        }
        first.setValue(biasedValue);
    }

    /**
     * Perform a forward pass and return predicted class indices.
     */
    public List<Integer> predict() {
        ArrayList<Integer> classLabels = forwardCalculation();
        clear();
        return classLabels;
    }

    /**
     * Perform a forward pass through the computational graph.
     * @return A list of predicted class indices.
     */
    public ArrayList<Integer> forwardCalculation() {
        ArrayList<ComputationalNode> sortedNodes = topologicalSort();
        if (sortedNodes.isEmpty()) return new ArrayList<>();
        ComputationalNode outputNode = sortedNodes.get(0);
        while (sortedNodes.size() > 1) {
            ComputationalNode currentNode = sortedNodes.remove(sortedNodes.size() - 1);
            List<ComputationalNode> children = nodeMap.get(currentNode);
            if (children != null) {
                for (ComputationalNode child : children) {
                    if (child.getValue() == null) {
                        if (child.getFunction() != null) {
                            Function function = child.getFunction();
                            Tensor currentValue = currentNode.getValue();
                            if (currentValue != null) {
                                child.setValue(function.calculate(currentValue));
                            }
                        } else {
                            if (currentNode.isBiased()) {
                                getBiased(currentNode);
                            }
                            child.setValue(currentNode.getValue());
                        }
                    } else {
                        if (child.getFunction() == null && child.getOperator() != null) {
                            switch (child.getOperator()) {
                                case "*": {
                                    if (currentNode.isBiased()) {
                                        getBiased(currentNode);
                                    }
                                    Tensor childValue = child.getValue();
                                    Tensor currentValue = currentNode.getValue();
                                    if (childValue != null && currentValue != null) {
                                        if (childValue.getShape()[1] == currentValue.getShape()[0]) {
                                            child.setValue(childValue.dot(currentValue));
                                        } else {
                                            child.setValue(currentValue.dot(childValue));
                                        }
                                    }
                                    break;
                                }
                                case "+": {
                                    Tensor result = child.getValue();
                                    Tensor currentValue = currentNode.getValue();
                                    if (result != null && currentValue != null) {
                                        child.setValue(result.add(currentValue));
                                    }
                                    break;
                                }
                                case "-": {
                                    Tensor result = child.getValue();
                                    Tensor currentValue = currentNode.getValue();
                                    if (result != null && currentValue != null) {
                                        child.setValue(result.subtract(currentValue));
                                    }
                                    break;
                                }
                                default:
                                    throw new IllegalArgumentException("Unsupported operator: " + child.getOperator());
                            }
                        }
                    }
                }
            }
        }
        ArrayList<Integer> classLabelIndices = new ArrayList<>();
        Tensor outputValue = outputNode.getValue();
        if (outputValue != null) {
            int rows = outputValue.getShape()[0];
            int cols = outputValue.getShape()[1];
            for (int i = 0; i < rows; i++) {
                double maxVal = Double.NEGATIVE_INFINITY;
                int labelIndex = -1;
                for (int j = 0; j < cols; j++) {
                    double val = outputValue.get(new int[]{i, j});
                    if (maxVal < val) {
                        maxVal = val;
                        labelIndex = j;
                    }
                }
                classLabelIndices.add(labelIndex);
            }
        }
        return classLabelIndices;
    }
}