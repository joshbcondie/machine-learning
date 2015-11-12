public class InstanceBasedLearner extends SupervisedLearner {

	private int neighborCount = 1;
	private boolean weightDistances = false;
	private Matrix m_features;
	private Matrix m_labels;
	private double[] minimums;
	private double[] maximums;

	private double[] normalize(double[] values, Matrix features) {
		double[] result = new double[values.length];
		for (int i = 0; i < values.length; i++) {
			if (features.valueCount(i) == 0)
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

		for (int i = 0; i < features.rows(); i++) {
			for (int j = 0; j < features.cols(); j++) {
				if (features.row(i)[j] < minimums[j])
					minimums[j] = features.row(i)[j];
				if (features.row(i)[j] > maximums[j])
					maximums[j] = features.row(i)[j];
			}
		}

		for (int i = 0; i < m_features.rows(); i++)
			for (int j = 0; j < m_features.cols(); j++)
				if (m_features.valueCount(j) == 0)
					m_features.set(i, j, (features.get(i, j) - minimums[j])
							/ (maximums[j] - minimums[j]));
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {

		double[] normalizedFeatures = normalize(features, m_features);

		double[] distances = new double[m_features.rows()];
		double[] topDistances = new double[neighborCount];
		for (int i = 0; i < topDistances.length; i++)
			topDistances[i] = Double.MAX_VALUE;
		int[] topInstances = new int[neighborCount];
		for (int i = 0; i < topInstances.length; i++)
			topInstances[i] = -1;

		for (int i = 0; i < distances.length; i++) {
			for (int j = 0; j < normalizedFeatures.length; j++) {
				if (m_features.valueCount(j) == 0)
					distances[i] += (m_features.get(i, j) - normalizedFeatures[j])
							* (m_features.get(i, j) - normalizedFeatures[j]);
				else if (m_features.get(i, j) != normalizedFeatures[j])
					distances[i]++;
			}

			for (int j = 0; j < neighborCount; j++) {
				if (distances[i] < topDistances[j]) {
					for (int k = neighborCount - 1; k > i; k--) {
						topDistances[k] = topDistances[k - 1];
						topInstances[k] = topInstances[k - 1];
					}

					topDistances[j] = distances[i];
					topInstances[j] = i;
					break;
				}
			}
		}

		if (weightDistances && topDistances[0] == 0)
			labels[0] = m_labels.row(topInstances[0])[0];
		else if (m_labels.valueCount(0) == 0) {
			double sum = 0;
			for (int i = 0; i < topInstances.length; i++) {
				if (weightDistances)
					sum += m_labels.row(topInstances[i])[0]
							/ (topDistances[i] * topDistances[i]);
				else
					sum += m_labels.row(topInstances[i])[0];
			}
			if (weightDistances) {
				double normalizer = 0;
				for (int i = 0; i < topDistances.length; i++)
					normalizer += 1 / (topDistances[i] * topDistances[i]);
				labels[0] = sum / normalizer;
			} else
				labels[0] = sum / topInstances.length;
		} else {
			double[] histogram = new double[m_labels.valueCount(0)];
			for (int i = 0; i < topInstances.length; i++) {
				if (weightDistances)
					histogram[(int) m_labels.row(topInstances[i])[0]] += 1 / (topDistances[i] * topDistances[i]);
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
