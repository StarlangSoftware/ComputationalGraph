import ComputationalGraph.*;
import org.junit.Test;
import Math.*;

import java.util.ArrayList;
import java.util.Random;

import static org.junit.Assert.*;

public class ComputationalGraphTest {

    @Test
    public void test1() throws MatrixDimensionMismatch, MatrixRowColumnMismatch {
        ComputationalGraph graph = new ComputationalGraph();
        // input matrix with bias.
        ComputationalNode input = new ComputationalNode(false, '*');
        Matrix m1 = new Matrix(3, 4, -0.01, 0.01, new Random(1));
        ComputationalNode w1 = new ComputationalNode(m1, '*');
        ComputationalNode a1 = graph.addEdge(input, w1);
        ComputationalNode a1Sigmoid = graph.addEdge(a1, FunctionType.SIGMOID);
        Matrix m2 = new Matrix(4, 5, -0.01, 0.01, new Random(2));
        ComputationalNode w2 = new ComputationalNode(m2, '*');
        ComputationalNode a2 = graph.addEdge(a1Sigmoid, w2);
        ComputationalNode output = graph.addEdge(a2, FunctionType.SOFTMAX);
        int epoch = 1000;
        double learningRate = 0.01;
        ArrayList<Integer> classList = new ArrayList<>();
        for (int i = 0; i < epoch; i++) {
            // Creating input vector.
            input.setValue(new Matrix(1, 3, 0, 100, new Random(1)));
            graph.forwardCalculation();
            classList.add(3);
            graph.backpropagation(learningRate, classList);
            classList.clear();
        }
        input.setValue(new Matrix(1, 3, 0, 100, new Random(1)));
        int classLabel = graph.forwardCalculation().get(0);
        assertEquals(3, classLabel);
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
