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
	final static double momentum = 0.9;
	final static double validationSetSize = 0.25;
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

		double[][] deltas = new double[hiddenLayerCount + 1][];
		for (int i = 0; i < deltas.length; i++) {
			if (i == deltas.length - 1)
				deltas[i] = new double[outputWeights.length];
			else
				deltas[i] = new double[hiddenWeights[i].length];
			for (int j = 0; j < deltas[i].length; j++)
				deltas[i][j] = 0;
		}

		double[][][] lastWeightChanges = new double[hiddenLayerCount + 1][][];
		for (int i = 0; i < lastWeightChanges.length; i++) {
			if (i == lastWeightChanges.length - 1)
				lastWeightChanges[i] = new double[outputWeights.length][];
			else
				lastWeightChanges[i] = new double[hiddenWeights[i].length][];
			for (int j = 0; j < lastWeightChanges[i].length; j++) {
				if (i == lastWeightChanges.length - 1)
					lastWeightChanges[i][j] = new double[outputWeights[j].length];
				else
					lastWeightChanges[i][j] = new double[hiddenWeights[i][j].length];
				for (int k = 0; k < lastWeightChanges[i][j].length; k++)
					lastWeightChanges[i][j][k] = 0;
			}
		}

		// printWeights();

		double[] guessedLabels = new double[labels.cols()];
		double wrongGuesses = Integer.MAX_VALUE;
		ArrayList<Double> epochAccuracies = new ArrayList<>();
		int validationSetNumber = (int) (validationSetSize * features.rows());
		Matrix validationFeatures = new Matrix(features, 0, 0,
				validationSetNumber, features.cols());
		Matrix trainingFeatures = new Matrix(features, validationSetNumber, 0,
				features.rows() - validationSetNumber, features.cols());
		Matrix validationLabels = new Matrix(labels, 0, 0, validationSetNumber,
				labels.cols());
		Matrix trainingLabels = new Matrix(labels, validationSetNumber, 0,
				labels.rows() - validationSetNumber, labels.cols());

		do {
			wrongGuesses = 0;

			trainingFeatures.shuffle(random, trainingLabels);

			for (int i = 0; i < trainingFeatures.rows(); i++) {
				predict(trainingFeatures.row(i), guessedLabels);

				for (int j = 0; j < trainingLabels.cols(); j++) {
					deltas[deltas.length - 1][j] = (trainingLabels.get(i, j) - unroundedOutputs[j])
							* unroundedOutputs[j] * (1 - unroundedOutputs[j]);
				}

				for (int j = hiddenLayerCount - 1; j >= 0; j--) {
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

				for (int j = 0; j < trainingLabels.cols(); j++) {
					for (int k = 0; k < outputWeights[j].length - 1; k++) {
						outputWeights[j][k] += learningRate
								* inputs[inputs.length - 1][k]
								* deltas[deltas.length - 1][j]
								+ momentum
								* lastWeightChanges[lastWeightChanges.length - 1][j][k];
						lastWeightChanges[lastWeightChanges.length - 1][j][k] = learningRate
								* inputs[inputs.length - 1][k]
								* deltas[deltas.length - 1][j]
								+ momentum
								* lastWeightChanges[lastWeightChanges.length - 1][j][k];
					}
					// update bias
					outputWeights[j][outputWeights[j].length - 1] += learningRate
							* deltas[deltas.length - 1][j]
							+ momentum
							* lastWeightChanges[lastWeightChanges.length - 1][j][outputWeights[j].length - 1];
					lastWeightChanges[lastWeightChanges.length - 1][j][outputWeights[j].length - 1] = learningRate
							* deltas[deltas.length - 1][j]
							+ momentum
							* lastWeightChanges[lastWeightChanges.length - 1][j][outputWeights[j].length - 1];
				}

				for (int j = hiddenLayerCount - 1; j >= 0; j--) {
					for (int k = 0; k < hiddenWeights[j].length; k++) {
						for (int m = 0; m < hiddenWeights[j][k].length - 1; m++) {
							hiddenWeights[j][k][m] += learningRate
									* inputs[j][m] * deltas[j][k] + momentum
									* lastWeightChanges[j][k][m];
							lastWeightChanges[j][k][m] = learningRate
									* inputs[j][m] * deltas[j][k] + momentum
									* lastWeightChanges[j][k][m];
						}
						// update bias
						hiddenWeights[j][k][hiddenWeights[j][k].length - 1] += learningRate
								* deltas[j][k]
								+ momentum
								* lastWeightChanges[j][k][hiddenWeights[j][k].length - 1];
						lastWeightChanges[j][k][hiddenWeights[j][k].length - 1] = learningRate
								* deltas[j][k]
								+ momentum
								* lastWeightChanges[j][k][hiddenWeights[j][k].length - 1];
					}
				}

				// System.out.println("Error values:");
				// for (int a = 0; a < deltas.length; a++)
				// for (int j = 0; j < deltas[a].length; j++)
				// System.out.println(deltas[a][j]);
			}

			for (int i = 0; i < validationFeatures.rows(); i++) {
				predict(validationFeatures.row(i), guessedLabels);

				for (int j = 0; j < validationLabels.cols(); j++) {
					wrongGuesses += Math.abs(validationLabels.get(i, j)
							- guessedLabels[j]);
				}
			}

			epochAccuracies.add((validationFeatures.rows() - wrongGuesses)
					/ validationFeatures.rows());
			// printWeights();
			System.out.println((validationFeatures.rows() - wrongGuesses)
					/ validationFeatures.rows());
		} while (epochAccuracies.size() < 6
				|| epochAccuracies.get(epochAccuracies.size() - 1)
						- epochAccuracies.get(epochAccuracies.size() - 6) > improvementThreshold);

		System.out.println(epochAccuracies.size() + " epochs");
	}

	public void printWeights() {
		System.out.println("Weights:");
		for (int i = 0; i < hiddenWeights.length; i++) {
			for (int j = 0; j < hiddenWeights[i].length; j++) {
				for (int k = 0; k < hiddenWeights[i][j].length; k++)
					System.out.print(String.format("%30s",
							hiddenWeights[i][j][k]));
				System.out.println();
			}
			System.out.println();
		}

		for (int i = 0; i < outputWeights.length; i++) {
			for (int j = 0; j < outputWeights[i].length; j++)
				System.out.print(String.format("%30s", outputWeights[i][j]));
			System.out.println();
		}

		System.out.println();
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
		// System.out.println("Predicted output:");
		for (int i = 0; i < labels.length; i++) {
			double sum = 0;
			for (int j = 0; j < inputs[inputs.length - 1].length; j++)
				sum += inputs[inputs.length - 1][j] * outputWeights[i][j];
			sum += outputWeights[i][outputWeights[i].length - 1];
			// labels[i] = sum > 0 ? 1 : 0;
			unroundedOutputs[i] = 1 / (1 + Math.pow(Math.E, -sum));
			// System.out.println(unroundedOutputs[i]);
			labels[i] = Math.round(unroundedOutputs[i]);
		}
	}

}
