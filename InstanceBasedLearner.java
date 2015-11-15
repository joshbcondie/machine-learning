public class InstanceBasedLearner extends SupervisedLearner {

	private int neighborCount = 1;
	private boolean weightDistances = false;
	private Matrix m_features;
	private Matrix m_labels;
	private double[] minimums;
	private double[] maximums;
	private boolean useHVDM = true;
	private double[][][] probabilities;

	private double[] normalize(double[] values, Matrix features) {
		double[] result = new double[values.length];
		for (int i = 0; i < values.length; i++) {
			if (features.valueCount(i) == 0 && values[i] != Matrix.MISSING)
				result[i] = (values[i] - minimums[i])
						/ (maximums[i] - minimums[i]);
			else
				result[i] = values[i];
		}
		return result;
	}

	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		m_features = new Matrix(features, 0, 0, features.rows(),
				features.cols());
		m_labels = new Matrix(labels, 0, 0, labels.rows(), labels.cols());
		minimums = new double[features.cols()];
		maximums = new double[features.cols()];
		if (m_labels.valueCount(0) == 0)
			useHVDM = false;
		if (useHVDM) {
			probabilities = new double[features.cols()][][];
			for (int i = 0; i < features.cols(); i++) {
				probabilities[i] = new double[m_features.valueCount(i)][];
				for (int j = 0; j < m_features.valueCount(i); j++)
					probabilities[i][j] = new double[m_labels.valueCount(0)];
			}
		}

		for (int i = 0; i < features.rows(); i++) {
			for (int j = 0; j < features.cols(); j++) {
				if (features.get(i, j) == Matrix.MISSING)
					continue;
				if (features.get(i, j) < minimums[j])
					minimums[j] = features.row(i)[j];
				if (features.get(i, j) > maximums[j])
					maximums[j] = features.row(i)[j];
				if (useHVDM) {
					if (m_features.valueCount(j) > 0) {
						probabilities[j][(int) features.get(i, j)][(int) labels
								.get(i, 0)]++;
					}
				}
			}
		}

		for (int i = 0; i < m_features.rows(); i++)
			for (int j = 0; j < m_features.cols(); j++)
				if (m_features.valueCount(j) == 0
						&& features.get(i, j) != Matrix.MISSING)
					m_features.set(i, j, (features.get(i, j) - minimums[j])
							/ (maximums[j] - minimums[j]));

		if (useHVDM) {
			for (int i = 0; i < probabilities.length; i++) {
				for (int j = 0; j < probabilities[i].length; j++) {
					int sum = 0;
					for (int k = 0; k < probabilities[i][j].length; k++)
						sum += probabilities[i][j][k];
					for (int k = 0; k < probabilities[i][j].length; k++) {
						if (sum == 0)
							probabilities[i][j][k] = 1.0 / m_labels
									.valueCount(0);
						else
							probabilities[i][j][k] /= sum;
					}
				}
			}
		}
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {

		double[] normalizedFeatures = normalize(features, m_features);

		double[] squaredDistances = new double[m_features.rows()];
		double[] topSquaredDistances = new double[neighborCount];
		for (int i = 0; i < topSquaredDistances.length; i++)
			topSquaredDistances[i] = Double.MAX_VALUE;
		int[] topInstances = new int[neighborCount];
		for (int i = 0; i < topInstances.length; i++)
			topInstances[i] = -1;

		for (int i = 0; i < squaredDistances.length; i++) {
			for (int j = 0; j < normalizedFeatures.length; j++) {
				if (m_features.get(i, j) == Matrix.MISSING
						|| normalizedFeatures[j] == Matrix.MISSING)
					squaredDistances[i]++;
				else if (m_features.valueCount(j) == 0)
					squaredDistances[i] += (m_features.get(i, j) - normalizedFeatures[j])
							* (m_features.get(i, j) - normalizedFeatures[j]);
				else {
					if (useHVDM) {
						for (int k = 0; k < m_labels.valueCount(0); k++) {
							squaredDistances[i] += (probabilities[j][(int) m_features
									.get(i, j)][k] - probabilities[j][(int) normalizedFeatures[j]][k])
									* (probabilities[j][(int) m_features.get(i,
											j)][k] - probabilities[j][(int) normalizedFeatures[j]][k]);
						}
					} else if (m_features.get(i, j) != normalizedFeatures[j])
						squaredDistances[i]++;
				}
			}

			for (int j = 0; j < neighborCount; j++) {
				if (squaredDistances[i] < topSquaredDistances[j]) {
					for (int k = neighborCount - 1; k > i; k--) {
						topSquaredDistances[k] = topSquaredDistances[k - 1];
						topInstances[k] = topInstances[k - 1];
					}

					topSquaredDistances[j] = squaredDistances[i];
					topInstances[j] = i;
					break;
				}
			}
		}

		if (weightDistances && topSquaredDistances[0] == 0)
			labels[0] = m_labels.row(topInstances[0])[0];
		else if (m_labels.valueCount(0) == 0) {
			double sum = 0;
			for (int i = 0; i < topInstances.length; i++) {
				if (weightDistances)
					sum += m_labels.row(topInstances[i])[0]
							/ topSquaredDistances[i];
				else
					sum += m_labels.row(topInstances[i])[0];
			}
			if (weightDistances) {
				double normalizer = 0;
				for (int i = 0; i < topSquaredDistances.length; i++)
					normalizer += 1 / topSquaredDistances[i];
				labels[0] = sum / normalizer;
			} else
				labels[0] = sum / topInstances.length;
		} else {
			double[] histogram = new double[m_labels.valueCount(0)];
			for (int i = 0; i < topInstances.length; i++) {
				if (weightDistances)
					histogram[(int) m_labels.row(topInstances[i])[0]] += 1 / topSquaredDistances[i];
				else
					histogram[(int) m_labels.row(topInstances[i])[0]]++;
			}

			double max = 0;
			int maxIndex = -1;
			for (int i = 0; i < histogram.length; i++) {
				if (histogram[i] > max) {
					max = histogram[i];
					maxIndex = i;
				}
			}

			labels[0] = maxIndex;
		}
	}
}
