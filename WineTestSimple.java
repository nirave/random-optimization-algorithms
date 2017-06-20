package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;
import util.linalg.DenseVector;
//import org.apache.commons;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer
 * or more than 15 rings.
 *
 * @author Hannah Lau
 * @version 1.0
 */

public class WineTestSimple {
    private static int rowCount = 1599;

    private static Instance[] instances = initializeInstances();
    private static Instance[] train_set = Arrays.copyOfRange(instances, 0, (int)(rowCount*0.75));
    private static Instance[] test_set = Arrays.copyOfRange(instances, (int)(rowCount*0.75), rowCount);

    //private static int inputLayer = 11, hiddenLayer = 5, outputLayer = 9, trainingIterations = 20000;
    private static int inputLayer = 11, hiddenLayer1 = 15, hiddenLayer2 = 15, outputLayer = 9, trainingIterations = 1000;

    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    //private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[1];

    //private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String[] oaNames = {"RHC"};

    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                    //new int[] {inputLayer, hiddenLayer, outputLayer});
                    new int[] {inputLayer, hiddenLayer1, hiddenLayer2, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        //Uncomment out this line for Randomized HIll Climbing
        oa[0] = new RandomizedHillClimbing(nnop[0]);

        //Uncomment out this line for Simulating Annealing
        //oa[0] = new SimulatedAnnealing(1E11, .95, nnop[1]);

        //Uncomment out this line for Genetic Algorithms
        //oa[0] = new StandardGeneticAlgorithm(200, 100, 10, nnop[0]);

        //
        //oa[0] = new RandomizedHillClimbing(nnop[0]);
        //oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        //oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

        for(int i = 0; i < oa.length; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            train(oa[i], networks[i], oaNames[i]); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            //Training set

            double predicted, actual;
            start = System.nanoTime();
            for(int j = 0; j < train_set.length; j++) {
                networks[i].setInputValues(instances[j].getData());
                networks[i].run();

                predicted = instances[j].getLabel().getData().argMax();
                actual = networks[i].getOutputValues().argMax();
                System.out.println("actual " + actual);

                //double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
                if (predicted == actual) {
                    correct++;
                } else {
                    incorrect++;
                }

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            results +=  "\nTraining Results for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                    + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";

            //Testing set
            correct = 0;
            incorrect = 0;
            //BackPropagationNetwork cloned = new BackPropagationNetwork(networks[i]);
            start = System.nanoTime();
            for(int j = 0; j < test_set.length; j++) {
                networks[i].setInputValues(test_set[j].getData());
                networks[i].run();

                predicted = instances[j].getLabel().getData().argMax();
                actual = networks[i].getOutputValues().argMax();
                System.out.println("actual " + actual);

                //double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
                if (predicted == actual) {
                    correct++;
                } else {
                    incorrect++;
                }

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            results +=  "\nTest Results for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                    + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
        }

        System.out.println(results);
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(network.getOutputValues()));
                //error = 1;
                error += measure.value(output, example);
            }

            System.out.println(df.format(error));
        }
    }

    public static int indexOfLargest(double array[])
    {
        int index = 0;
        double max_value = 0;

        for (int i = 0; i < array.length; i++) {
            if (array[i] > max_value) {
                index = i;
                max_value = array[i];
            }
        }

        return index;
    }

    private static Instance[] initializeInstances() {

        double[][][] attributes = new double[1599][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("./winequality-red.csv")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(";");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[11]; // 11 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < 11; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);

            // Read the digit 0-9 from the attribute array that was read from the csv
            int c = (int) attributes[i][1][0];

            int nClasses = 9;
            // Create a double array of length 10, all values are initialized to 0
            double[] classes = new double[nClasses];

            // Set the i'th index to 1.0
            classes[c] = 1.0;
            instances[i].setLabel(new Instance(classes));

        }

        return instances;
    }
}
