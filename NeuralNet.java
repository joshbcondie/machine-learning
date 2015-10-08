// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

import java.util.ArrayList;
import java.util.Random;

public class NeuralNet extends SupervisedLearner {

	static int hiddenLayerCount = 2;
	final static int neuronsPerHiddenLayer = 2;
	final static double learningRate = 0.1;
	final static double improvementThreshold = 0.005;
	double[][][] hiddenWeights;
	double[][] outputWeights;
	double[][] inputs;
	double[] unroundedOutputs;
	Random random;

	public NeuralNet(Random random) {
		this.random = random;
	}

	public void train(Matrix features, Matrix labels) throws Exception {

		hiddenWeights = new double[hiddenLayerCount][][];
		for (int i = 0; i < hiddenWeights.length; i++) {
			hiddenWeights[i] = new double[neuronsPerHiddenLayer][];
			for (int j = 0; j < hiddenWeights[i].length; j++) {
				if (i == 0)
					hiddenWeights[i][j] = new double[features.cols() + 1];
				else
					hiddenWeights[i][j] = new double[hiddenWeights[i - 1].length + 1];
				for (int k = 0; k < hiddenWeights[i][j].length; k++) {
					hiddenWeights[i][j][k] = random.nextDouble()
							* (random.nextInt(2) == 0 ? -1 : 1);
				}
			}
		}

		outputWeights = new double[labels.cols()][];
		for (int i = 0; i < outputWeights.length; i++) {
			if (hiddenLayerCount == 0)
				outputWeights[i] = new double[features.cols() + 1];
			else
				outputWeights[i] = new double[hiddenWeights[hiddenWeights.length - 1].length + 1];
			for (int j = 0; j < outputWeights[i].length; j++) {
				outputWeights[i][j] = random.nextDouble()
						* (random.nextInt(2) == 0 ? -1 : 1);
			}
		}

		double[] guessedLabels = new double[labels.cols()];
		double wrongGuesses = Integer.MAX_VALUE;
		ArrayList<Double> epochAccuracies = new ArrayList<>();

		do {
			wrongGuesses = 0;

			features.shuffle(random, labels);

			double[][] deltas = new double[hiddenLayerCount + 1][];
			for (int i = 0; i < features.rows(); i++) {
				predict(features.row(i), guessedLabels);

				deltas[deltas.length - 1] = new double[outputWeights.length];
				for (int j = 0; j < labels.cols(); j++) {
					wrongGuesses += Math.abs(labels.get(i, j)
							- guessedLabels[j]);
					deltas[deltas.length - 1][j] = (labels.get(i, j) - unroundedOutputs[j])
							* unroundedOutputs[j] * (1 - unroundedOutputs[j]);
				}

				for (int j = hiddenLayerCount - 1; j >= 0; j--) {
					deltas[j] = new double[hiddenWeights[j].length];
					for (int k = 0; k < hiddenWeights[j].length; k++) {

						deltas[j][k] = 0;

						if (j == hiddenLayerCount - 1) {
							for (int m = 0; m < outputWeights.length; m++)
								deltas[j][k] += deltas[j + 1][m]
										* outputWeights[m][k];
						} else {
							for (int m = 0; m < hiddenWeights[j + 1].length; m++)
								deltas[j][k] += deltas[j + 1][m]
										* hiddenWeights[j + 1][m][k];
						}

						deltas[j][k] *= inputs[j + 1][k]
								* (1 - inputs[j + 1][k]);
					}
				}

				for (int j = 0; j < labels.cols(); j++) {
					for (int k = 0; k < outputWeights[j].length - 1; k++)
						outputWeights[j][k] += learningRate
								* inputs[inputs.length - 1][k]
								* deltas[deltas.length - 1][j];
					// update bias
					outputWeights[j][outputWeights[j].length - 1] += learningRate
							* deltas[deltas.length - 1][j];
				}

				for (int j = hiddenLayerCount - 1; j >= 0; j--) {
					for (int k = 0; k < hiddenWeights[j].length; k++) {
						for (int m = 0; m < hiddenWeights[j][k].length - 1; m++)
							hiddenWeights[j][k][m] += learningRate
									* inputs[j + 1][k] * deltas[j][k];
						// update bias
						hiddenWeights[j][k][hiddenWeights[j][k].length - 1] += learningRate
								* deltas[j][k];
					}
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

		inputs = new double[hiddenLayerCount + 1][];
		inputs[0] = new double[features.length];
		for (int i = 0; i < features.length; i++)
			inputs[0][i] = features[i];

		for (int i = 0; i < hiddenLayerCount; i++) {
			inputs[i + 1] = new double[hiddenWeights[i].length];
			for (int j = 0; j < hiddenWeights[i].length; j++) {
				double sum = 0;
				for (int k = 0; k < inputs[i].length; k++)
					sum += inputs[i][k] * hiddenWeights[i][j][k];
				sum += hiddenWeights[i][j][hiddenWeights[i][j].length - 1];
				inputs[i + 1][j] = 1 / (1 + Math.pow(Math.E, -sum));
			}
		}
		unroundedOutputs = new double[outputWeights.length];
		for (int i = 0; i < labels.length; i++) {
			double sum = 0;
			for (int j = 0; j < inputs[inputs.length - 1].length; j++)
				sum += inputs[inputs.length - 1][j] * outputWeights[i][j];
			sum += outputWeights[i][outputWeights[i].length - 1];
			// labels[i] = sum > 0 ? 1 : 0;
			unroundedOutputs[i] = 1 / (1 + Math.pow(Math.E, -sum));
			labels[i] = Math.round(unroundedOutputs[i]);
		}
	}

}
