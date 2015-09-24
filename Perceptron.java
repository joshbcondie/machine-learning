// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

import java.util.ArrayList;
import java.util.Random;

public class Perceptron extends SupervisedLearner {

	final static double learningRate = 0.1;
	final static double improvementThreshold = 0.005;
	double[][] weights;
	Random random;

	public Perceptron(Random random) {
		this.random = random;
	}

	public void train(Matrix features, Matrix labels) throws Exception {
		weights = new double[labels.cols()][];
		for (int i = 0; i < weights.length; i++) {
			weights[i] = new double[features.cols() + 1];
			for (int j = 0; j < weights[i].length; j++) {
				weights[i][j] = random.nextDouble()
						* (random.nextInt(2) == 0 ? -1 : 1);
			}
		}

		double[] guessedLabels = new double[labels.cols()];
		double wrongGuesses = Integer.MAX_VALUE;
		ArrayList<Double> epochAccuracies = new ArrayList<>();

		do {
			wrongGuesses = 0;

			features.shuffle(random, labels);

			for (int i = 0; i < features.rows(); i++) {
				predict(features.row(i), guessedLabels);
				for (int j = 0; j < labels.cols(); j++) {
					wrongGuesses += Math.abs(labels.get(i, j)
							- guessedLabels[j]);
					for (int k = 0; k < features.cols(); k++)
						weights[j][k] += (labels.get(i, j) - guessedLabels[j])
								* learningRate * features.get(i, k);
					weights[j][weights[j].length - 1] += (labels.get(i, j) - guessedLabels[j])
							* learningRate;
				}
			}
			epochAccuracies.add((features.rows() - wrongGuesses)
					/ features.rows());
			System.out.println((features.rows() - wrongGuesses)
					/ features.rows());
		} while (epochAccuracies.size() < 6
				|| epochAccuracies.get(epochAccuracies.size() - 1)
						- epochAccuracies.get(epochAccuracies.size() - 6) > improvementThreshold);

		System.out.println(epochAccuracies.size() + " epochs");
	}

	public void predict(double[] features, double[] labels) throws Exception {
		for (int i = 0; i < labels.length; i++) {
			double sum = 0;
			for (int j = 0; j < features.length; j++)
				sum += features[j] * weights[i][j];
			sum += weights[i][weights[i].length - 1];
			labels[i] = sum > 0 ? 1 : 0;
		}
	}

}
