import java.io.FileNotFoundException;
import java.util.Random;

public class Clustering {

	private static final boolean normalize = false;
	private static final boolean shuffle = false;
	private static final boolean randomSeed = false;
	private static long seed = 555;
	private static final String fileName = "sponge.arff";
	private static final int k = 4;
	private static final boolean useEuclideanDistance = true;
	private static Matrix data;

	public static void main(String[] args) throws FileNotFoundException,
			Exception {
		data = new Matrix();
		data.loadArff(fileName);
		if (normalize) {
			System.out.println("Using normalized data\n");
			data.normalize();
		}

		if (shuffle) {
			if (randomSeed) {
				Random random = new Random();
				seed = random.nextLong();
			}
			System.out.println("Random seed: " + seed);
			data.shuffle(new Random(seed));
		}

		double[][] centroids = new double[k][];
		boolean clustersChanged = true;
		int[] clusters = new int[data.rows()];

		for (int i = 0; i < k; i++) {
			clusters[i] = i;
			centroids[i] = new double[data.cols()];
		}

		for (int i = k; i < data.rows(); i++)
			clusters[i] = -1;

		int iteration = 1;

		while (clustersChanged) {

			clustersChanged = false;

			System.out.println("***************");
			System.out.println("Iteration " + iteration);
			System.out.println("***************");
			System.out.println("Computing Centroids:");
			for (int i = 0; i < k; i++) {
				int[] numPoints = new int[data.cols()];
				double[] sum = new double[data.cols()];
				int[][] histogram = new int[data.cols()][];
				for (int c = 0; c < histogram.length; c++)
					histogram[c] = new int[data.valueCount(c)];

				for (int r = 0; r < data.rows(); r++) {
					if (clusters[r] == i) {
						for (int c = 0; c < data.cols(); c++) {
							if (data.get(r, c) != Matrix.MISSING) {
								numPoints[c]++;
								if (data.valueCount(c) == 0)
									sum[c] += data.get(r, c);
								else
									histogram[c][(int) data.get(r, c)]++;
							}
						}
					}
				}
				for (int c = 0; c < data.cols(); c++) {
					double lastValue = centroids[i][c];
					if (data.valueCount(c) == 0) {
						if (numPoints[c] > 0)
							centroids[i][c] = sum[c] / numPoints[c];
						else
							centroids[i][c] = Matrix.MISSING;
					} else {
						int max = 0;
						int maxIndex = -1;
						for (int j = 0; j < data.valueCount(c); j++) {
							if (histogram[c][j] > max) {
								max = histogram[c][j];
								maxIndex = j;
							}
						}
						if (maxIndex >= 0)
							centroids[i][c] = maxIndex;
						else
							centroids[i][c] = Matrix.MISSING;
					}
					if (centroids[i][c] != lastValue)
						clustersChanged = true;
				}
				System.out.print("Centroid " + i + " = ");
				for (int c = 0; c < data.cols(); c++) {
					if (centroids[i][c] == Matrix.MISSING)
						System.out.print("?");
					else if (data.valueCount(c) == 0)
						System.out.print(centroids[i][c]);
					else
						System.out.print(data.attrValue(c,
								(int) centroids[i][c]));
					if (c != data.cols() - 1)
						System.out.print(", ");
				}
				System.out.println();
			}

			System.out.println("Making Assignments");
			double sse = 0;
			int[] numInstances = new int[k];
			double[] sses = new double[k];

			for (int r = 0; r < data.rows(); r++) {
				double minSquaredDistance = Double.MAX_VALUE;
				int closestK = -1;
				for (int i = 0; i < k; i++) {
					double squaredDistance;
					if (useEuclideanDistance)
						squaredDistance = squaredDistance(data.row(r),
								centroids[i]);
					else
						squaredDistance = manhattanDistance(data.row(r),
								centroids[i])
								* manhattanDistance(data.row(r), centroids[i]);
					if (squaredDistance < minSquaredDistance) {
						minSquaredDistance = squaredDistance;
						closestK = i;
					}
				}
				clusters[r] = closestK;
				numInstances[closestK]++;
				sses[closestK] += minSquaredDistance;
				sse += minSquaredDistance;
				if (r % 10 == 0) {
					if (r > 0)
						System.out.println();
					System.out.print("\t");
				}
				System.out.print(r + "=" + closestK + " ");
			}
			System.out.println();
			for (int i = 0; i < k; i++) {
				System.out.println("Centroid " + i + ": numInstances = "
						+ numInstances[i] + "; SSE = " + sses[i]);
			}
			System.out.print("\nTotal SSE: " + sse + "\n\n");

			iteration++;
		}

		int[] nearestClusters = new int[centroids.length];
		for (int i = 0; i < nearestClusters.length; i++) {
			int nearestCluster = -1;
			double minSquaredDistance = Double.MAX_VALUE;
			for (int j = 0; j < nearestClusters.length; j++) {
				if (i == j)
					continue;
				double squaredDistance = squaredDistance(centroids[i],
						centroids[j]);
				if (squaredDistance < minSquaredDistance) {
					minSquaredDistance = squaredDistance;
					nearestCluster = j;
				}
			}
			nearestClusters[i] = nearestCluster;
		}

		double silhouetteCoefficient = 0;
		for (int i = 0; i < clusters.length; i++) {

			double a = 0;
			int aCount = 0;
			double b = 0;
			int bCount = 0;

			for (int j = 0; j < clusters.length; j++) {
				if (i == j)
					continue;
				if (clusters[i] == clusters[j]) {
					a += Math.sqrt(squaredDistance(data.row(i), data.row(j)));
					aCount++;
				} else if (nearestClusters[clusters[i]] == clusters[j]) {
					b += Math.sqrt(squaredDistance(data.row(i), data.row(j)));
					bCount++;
				}
			}
			if (aCount > 0)
				a /= aCount;
			if (bCount > 0)
				b /= bCount;
			silhouetteCoefficient += (b - a) / Math.max(a, b);
		}
		silhouetteCoefficient /= data.rows();
		System.out.println("Silhouette Coefficient: " + silhouetteCoefficient);
	}

	private static double squaredDistance(double[] a, double[] b) {
		double squaredDistance = 0;
		for (int c = 0; c < a.length; c++) {
			if (a[c] == Matrix.MISSING || b[c] == Matrix.MISSING)
				squaredDistance++;
			else {
				if (data.valueCount(c) == 0)
					squaredDistance += (a[c] - b[c]) * (a[c] - b[c]);
				else if (a[c] != b[c])
					squaredDistance++;
			}
		}
		return squaredDistance;
	}

	private static double manhattanDistance(double[] a, double[] b) {
		double distance = 0;
		for (int c = 0; c < a.length; c++) {
			if (a[c] == Matrix.MISSING || b[c] == Matrix.MISSING)
				distance++;
			else {
				if (data.valueCount(c) == 0)
					distance += Math.abs(a[c] - b[c]);
				else if (a[c] != b[c])
					distance++;
			}
		}
		return distance;
	}
}
