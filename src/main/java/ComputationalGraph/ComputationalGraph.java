package ComputationalGraph;

import Math.Tensor;
import java.util.List;
import java.util.Set;
import java.util.Map;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Deque;
import java.util.LinkedList;

public class ComputationalGraph {
    private Map<ComputationalNode, List<ComputationalNode>> node_map = new HashMap<>();
    private Map<ComputationalNode, List<ComputationalNode>> reverse_node_map = new HashMap<>();

    public ComputationalNode addEdge(ComputationalNode first, Object second, boolean isBiased) {
        ComputationalNode new_node;
    
        if (second instanceof FunctionType) {
            new_node = new ComputationalNode(false, isBiased, null, (FunctionType) second, null);
        }
        else if (second instanceof ComputationalNode) {
            new_node = new ComputationalNode(false, isBiased, ((ComputationalNode) second).getOperator(), null, null);
        }
        else {
            throw new IllegalArgumentException("Invalid type for 'second'. Must be a ComputationalNode or FunctionType.");
        }
    
        node_map.computeIfAbsent(first, k -> new ArrayList<>()).add(new_node);
        reverse_node_map.computeIfAbsent(new_node, k -> new ArrayList<>()).add(first);

        if (second instanceof ComputationalNode) {
            node_map.computeIfAbsent((ComputationalNode) second, k -> new ArrayList<>()).add(new_node);
            reverse_node_map.computeIfAbsent(new_node, k -> new ArrayList<>()).add((ComputationalNode) second);
        }
    
        return new_node;
    }
    

    private Deque<ComputationalNode> sortRecursive(ComputationalNode node, Set<ComputationalNode> visited) {
        /**
         * Recursive helper function to perform depth-first search for topological sorting.
         * :param node: The current node being processed.
         * :param visited: A set of visited nodes.
         * :return: A list representing the partial topological order.
         */
        Deque<ComputationalNode> queue = new LinkedList<>();
        visited.add(node);
        if (node_map.containsKey(node)) {
            for (ComputationalNode child : node_map.get(node)) {
                if (!visited.contains(child)) {
                    queue.addAll(sortRecursive(child, visited));
                }
            }
        }
        queue.offer(node);
        return queue;
    }

    public List<ComputationalNode> topologicalSort() {
        /**
         * Performs topological sorting on the computational graph.
         * :param node_map: A dictionary representing the graph (node -> list of child nodes).
         * :return: A list representing the topological order of the nodes.
         */
        Deque<ComputationalNode> sorted_list = new LinkedList<>();
        Set<ComputationalNode> visited = new HashSet<>();
        for (ComputationalNode node : node_map.keySet()) {
            if (!visited.contains(node)) {
                Deque<ComputationalNode> queue = sortRecursive(node, visited);
                while (!queue.isEmpty()) {
                    sorted_list.offerLast(queue.pollFirst());
                }
            }
        }
        return new ArrayList<>(sorted_list);
    }

    private void clearRecursive(Set<ComputationalNode> visited, ComputationalNode node) {
        /**
         * Recursive helper function to clear the values and gradients of nodes.
         */
        visited.add(node);
        if (!node.isLearnable()) {
            node.setValue(null);
        }
        node.setBackward(null);

        if (node_map.containsKey(node)) {
            for (ComputationalNode child : node_map.get(node)) {
                if (!visited.contains(child)) {
                    clearRecursive(visited, child);
                }
            }
        }
    }

    public void clear() {
        /**
         * Clears the values and gradients of all nodes in the graph.
         */
        Set<ComputationalNode> visited = new HashSet<>();
        for (ComputationalNode node : node_map.keySet()) {
            if (!visited.contains(node)) {
                clearRecursive(visited, node);
            }
        }
    }

    private void updateRecursive(Set<ComputationalNode> visited, ComputationalNode node) {
        /**
         * Recursive helper function to update the values of learnable nodes.
         */
        visited.add(node);
        if (node.isLearnable()) {
            node.updateValue();
        }

        if (node_map.containsKey(node)) {
            for (ComputationalNode child : node_map.get(node)) {
                if (!visited.contains(child)) {
                    updateRecursive(visited, child);
                }
            }
        }
    }

    public void updateValues() {
        /**
         * Updates the values of all learnable nodes in the graph.
         */
        Set<ComputationalNode> visited = new HashSet<>();
        for (ComputationalNode node : node_map.keySet()) {
            if (!visited.contains(node)) {
                updateRecursive(visited, node);
            }
        }
    }

    public Tensor calculateDerivative(ComputationalNode node, ComputationalNode child) {
        /**
         * Calculates the derivative of the child node with respect to the parent node.
         * :param node: Parent node.
         * :param child: Child node.
         * :return: The gradient tensor.
         */
        List<ComputationalNode> reverseChildren = reverse_node_map.get(child);
        if (reverseChildren == null || reverseChildren.isEmpty()) {
            return null; // Or handle this case appropriately
        }
        ComputationalNode left = reverseChildren.get(0);
        if (reverseChildren.size() == 1) {
            Function function = null;
            if (child.getFunctionType() == FunctionType.SIGMOID) {
                function = new Sigmoid();
            } else if (child.getFunctionType() == FunctionType.TANH) {
                function = new Tanh();
            } else if (child.getFunctionType() == FunctionType.RELU) {
                function = new ReLU();
            } else if (child.getFunctionType() == FunctionType.SOFTMAX) {
                function = new Softmax();
            } else {
                throw new IllegalArgumentException("Unsupported function type: " + child.getFunctionType());
            }
            Tensor backward = child.getBackward();
            Tensor derivative = function.derivative(child.getValue());
            if (backward != null && derivative != null) {
                return backward.multiply(derivative); // Optimized element-wise multiplication
            }
            return null;

        } else {
            ComputationalNode right = reverseChildren.get(1);
            if (child.getOperator() != null) {
                if (child.getOperator().equals("*")) {
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
                } else if (child.getOperator().equals("+")) {
                    return child.getBackward();
                } else if (child.getOperator().equals("-")) {
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

    public void calculateRMinusY(ComputationalNode output, double learning_rate, List<Integer> class_label_index) {
        /**
         * Computes the difference between the predicted and actual values (R - Y).
         * :param output: The output node of the computational graph.
         * :param learning_rate: The learning rate for gradient descent.
         * :param class_label_index: A list of true class labels (index of the correct class for each sample).
         */
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
                if (class_label_index.get(i) == j) {
                    backward.set(new int[]{i, j}, (1 - outputValue.get(new int[]{i, j})) * learning_rate);
                } else {
                    backward.set(new int[]{i, j}, (-outputValue.get(new int[]{i, j})) * learning_rate);
                }
            }
        }
        output.setBackward(backward);
    }

    public void backpropagation(double learning_rate, List<Integer> class_label_index) {
        /**
         * Performs backpropagation on the computational graph.
         * :param learning_rate: The learning rate for gradient descent.
         * :param class_label_index: The true class labels (as a list of integers).
         */
        List<ComputationalNode> sorted_nodes = topologicalSort();
        if (sorted_nodes.isEmpty()) return;
        ComputationalNode output_node = sorted_nodes.remove(0);
        calculateRMinusY(output_node, learning_rate, class_label_index);
        if (!sorted_nodes.isEmpty()) {
            sorted_nodes.remove(0).setBackward(output_node.getBackward());
        }
        while (!sorted_nodes.isEmpty()) {
            ComputationalNode node = sorted_nodes.remove(0);
            List<ComputationalNode> children = node_map.get(node);
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

    public void getBiased(ComputationalNode first) {
        /**
         * Add a bias term to the node's value by appending a column of ones.
         */
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
        Tensor biased_value = new Tensor(initialBiasedValueData, new int[]{rows, newCols});
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < originalCols; j++) {
                biased_value.set(new int[]{i, j}, firstValue.get(new int[]{i, j}));
            }
            biased_value.set(new int[]{i, originalCols}, 1.0);
        }
        first.setValue(biased_value);
    }

    public List<Integer> predict() {
        /**
         * Perform a forward pass and return predicted class indices.
         */
        List<Integer> class_labels = forwardCalculation();
        clear();
        return class_labels;
    }

    public List<Integer> forwardCalculation() {
        /**
         * Perform a forward pass through the computational graph.
         * Returns:
         * A list of predicted class indices.
         */
        List<ComputationalNode> sorted_nodes = topologicalSort();
        if (sorted_nodes.isEmpty()) return new ArrayList<>();
        ComputationalNode output_node = sorted_nodes.get(0);

        while (sorted_nodes.size() > 1) {
            ComputationalNode current_node = sorted_nodes.remove(sorted_nodes.size() - 1);
            List<ComputationalNode> children = node_map.get(current_node);
            if (children != null) {
                for (ComputationalNode child : children) {
                    if (child.getValue() == null) {
                        if (child.getFunctionType() != null) {
                            Function function = null;
                            if (child.getFunctionType() == FunctionType.SIGMOID) {
                                function = new Sigmoid();
                            } else if (child.getFunctionType() == FunctionType.TANH) {
                                function = new Tanh();
                            } else if (child.getFunctionType() == FunctionType.RELU) {
                                function = new ReLU();
                            } else if (child.getFunctionType() == FunctionType.SOFTMAX) {
                                function = new Softmax();
                            } else {
                                throw new IllegalArgumentException("Unsupported function type: " + child.getFunctionType());
                            }
                            Tensor currentValue = current_node.getValue();
                            if (currentValue != null) {
                                child.setValue(function.calculate(currentValue));
                            }
                        } else {
                            if (current_node.isBiased()) {
                                getBiased(current_node);
                            }
                            child.setValue(current_node.getValue());
                        }
                    } else {
                        if (child.getFunctionType() == null && child.getOperator() != null) {
                            if (child.getOperator().equals("*")) {
                                if (current_node.isBiased()) {
                                    getBiased(current_node);
                                }
                                Tensor childValue = child.getValue();
                                Tensor currentValue = current_node.getValue();
                                if (childValue != null && currentValue != null) {
                                    if (childValue.getShape()[1] == currentValue.getShape()[0]) {
                                        child.setValue(childValue.dot(currentValue));
                                    } else {
                                        child.setValue(currentValue.dot(childValue));
                                    }
                                }
                            } else if (child.getOperator().equals("+")) {
                                Tensor result = child.getValue();
                                Tensor currentValue = current_node.getValue();
                                if (result != null && currentValue != null) {
                                    child.setValue(result.add(currentValue));
                                }
                            } else if (child.getOperator().equals("-")) {
                                Tensor result = child.getValue();
                                Tensor currentValue = current_node.getValue();
                                if (result != null && currentValue != null) {
                                    child.setValue(result.subtract(currentValue));
                                }
                            } else {
                                throw new IllegalArgumentException("Unsupported operator: " + child.getOperator());
                            }
                        }
                    }
                }
            }
        }

        List<Integer> class_label_indices = new ArrayList<>();
        Tensor outputValue = output_node.getValue();
        if (outputValue != null) {
            int rows = outputValue.getShape()[0];
            int cols = outputValue.getShape()[1];
            for (int i = 0; i < rows; i++) {
                double max_val = Double.NEGATIVE_INFINITY;
                int label_index = -1;
                for (int j = 0; j < cols; j++) {
                    double val = outputValue.get(new int[]{i, j});
                    if (max_val < val) {
                        max_val = val;
                        label_index = j;
                    }
                }
                class_label_indices.add(label_index);
            }
        }

        return class_label_indices;
    }  
}