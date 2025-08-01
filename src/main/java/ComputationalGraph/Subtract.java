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
    public Tensor derivative(Tensor matrix) {
        return calculate(matrix);
    }
}
