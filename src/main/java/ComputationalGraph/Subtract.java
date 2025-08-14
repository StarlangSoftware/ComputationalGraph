package ComputationalGraph;

import java.io.Serializable;
import java.util.ArrayList;

import Math.*;

public class Subtract implements Function, Serializable {
    @Override
    public Tensor calculate(Tensor matrix) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> oldValues = (ArrayList<Double>) matrix.getData();
        for (Double oldValue : oldValues) {
            values.add(-oldValue);
        }
        return new Tensor(values, matrix.getShape());
    }

    @Override
    public Tensor derivative(Tensor matrix, Tensor backward) {
        ArrayList<Double> values = new ArrayList<>();
        int size = 1;
        for (int i = 0; i < matrix.getShape().length; i++) {
            size *= matrix.getShape()[i];
        }
        for (int i = 0; i < size; i++) {
            values.add(-1.0);
        }
        return backward.hadamardProduct(new Tensor(values, matrix.getShape()));
    }
}
