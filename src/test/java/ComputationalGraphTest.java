import ComputationalGraph.*;
import org.junit.Test;
import Math.*;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;

import static org.junit.Assert.*;

public class ComputationalGraphTest {

    private Matrix createInputMatrix(String[] instance) {
        Matrix matrix = new Matrix(1, instance.length - 1);
        for (int i = 0; i < instance.length - 1; i++) {
            matrix.setValue(0, i, Double.parseDouble(instance[i]));
        }
        return matrix;
    }

    @Test
    public void test1() throws MatrixDimensionMismatch, MatrixRowColumnMismatch, FileNotFoundException {
        HashMap<String, Integer> labelMap = new HashMap<>();
        ArrayList<String[]> instances = new ArrayList<>();
        ArrayList<String[]> testSet = new ArrayList<>();
        ArrayList<String[]> dataSet = new ArrayList<>();
        Scanner source = new Scanner(new File("iris.txt"));
        while (source.hasNextLine()) {
            String[] instance = source.nextLine().split(",");
            dataSet.add(instance);
            if (!labelMap.containsKey(instance[instance.length - 1])) {
                labelMap.put(instance[instance.length - 1], labelMap.size());
            }
        }
        source.close();
        Collections.shuffle(dataSet);
        for (int i = 0; i < dataSet.size(); i++) {
            if (i >= 135) {
                testSet.add(dataSet.get(i));
            } else {
                instances.add(dataSet.get(i));
            }
        }
        ComputationalGraph graph = new ComputationalGraph();
        ComputationalNode input = new ComputationalNode(false, '*');
        Matrix m1 = new Matrix(5, 4, -0.01, 0.01, new Random(1));
        ComputationalNode w1 = new ComputationalNode(m1, '*');
        ComputationalNode a1 = graph.addEdge(input, w1);
        ComputationalNode a1Sigmoid = graph.addEdge(a1, FunctionType.SIGMOID);
        Matrix m2 = new Matrix(5, 20, -0.01, 0.01, new Random(2));
        ComputationalNode w2 = new ComputationalNode(m2, '*');
        ComputationalNode a2 = graph.addEdge(a1Sigmoid, w2);
        ComputationalNode a2Sigmoid = graph.addEdge(a2, FunctionType.SIGMOID);
        Matrix m3 = new Matrix(21, labelMap.size(), -0.01, 0.01, new Random(3));
        ComputationalNode w3 = new ComputationalNode(m3, '*');
        ComputationalNode a3 = graph.addEdge(a2Sigmoid, w3);
        graph.addEdge(a3, FunctionType.SOFTMAX);
        int epoch = 1000;
        double learningRate = 0.01;
        ArrayList<Integer> classList = new ArrayList<>();
        for (int i = 0; i < epoch; i++) {
            Collections.shuffle(instances);
            for (String[] instance : instances) {
                input.setValue(createInputMatrix(instance));
                graph.forwardCalculation();
                classList.add(labelMap.get(instance[instance.length - 1]));
                graph.backpropagation(learningRate, classList);
                classList.clear();
            }
        }
        int count = 0;
        for (String[] strings : testSet) {
            input.setValue(createInputMatrix(strings));
            int classLabel = graph.predict().get(0);
            if (classLabel == labelMap.get(strings[strings.length - 1])) {
                count++;
            }
        }
        assertEquals(1.0, (count + 0.0) / 15, 0.001);
    }

    @Test
    public void test2() throws MatrixDimensionMismatch, MatrixRowColumnMismatch {
        ComputationalGraph graph = new ComputationalGraph();
        ComputationalNode a0 = new ComputationalNode(false, '+');
        ComputationalNode a1 = new ComputationalNode(true, '+');
        ComputationalNode a2 = graph.addEdge(a0, a1);
        ComputationalNode output = graph.addEdge(a2, FunctionType.SOFTMAX);
        a0.setValue(new Matrix(1, 3, 0, 100, new Random(1)));
        a1.setValue(new Matrix(1, 3, 0, 100, new Random(2)));
        graph.forwardCalculation();
        ArrayList<Integer> classList = new ArrayList<>();
        classList.add(1);
        graph.backpropagation(0.01, classList);
    }
}
