import Classification.Parameter.Parameter;

public class NeuralNetParameter extends Parameter {

    private final int epoch;
    private final double etaDecrease;
    private double learningRate;

    public NeuralNetParameter(int epoch, double etaDecrease, double learningRate, int seed) {
        super(seed);
        this.epoch = epoch;
        this.etaDecrease = etaDecrease;
        this.learningRate = learningRate;
    }

    public int getEpoch() {
        return epoch;
    }

    public double getEtaDecrease() {
        return etaDecrease;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate() {
        this.learningRate *= etaDecrease;
    }
}
