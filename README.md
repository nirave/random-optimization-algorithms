# random_optimization_algorithms

For full analysis, please see the pdf in the current repository.

The basics gist of the code is to run supervised learning against Red Wine and NFL statistics to determine quality and wins respectively.

This is for GATech's Graduate Machine Learning class, assignment 2

The following desribes how to run the Random Optmization algorithms from the assignment.

First, ABAGAIL must be downloaded from the following:

Copy the following 4 files into the src/opt/test directory and rebuild ABAGAIL.tar via ANT:
1. WineTestSimple.java
2. ContinuousPeaksTest.java (an alteration of the original ContinuousPeaksTest.java)
3. MaxKColoringTest.java (an alteration of the original MaxKColoringTest.java)
4. TravelingSalesmanProblem.java (an alteration of the original ContinuousPeaksTest.java)

In addition, copy winequality-red.csv to the directory current working directory.
## Neural Network
To run the Neural Network tests, run WineTestSimple via:
java -cp ABAGAIL.jar opt.test.WineTestSimple

Note that to run Randomized Hill Climbing, uncomment out line 63 and comment out lines 66,69
Note that to run Simulated Annealing, uncomment out line 66 and comment out lines 63,69
Note that to run Genetic Algorithm, uncomment out line 69 and comment out lines 63,66

Also note that line 48 should be altered to a string that represrents the name for printing purposes (e.g. "RHC")

## Continuous Peaks

To run Continuous Peaks Problem, run the modified ContinuousPeaksTest.java file via:
java -cp ABAGAIL.jar opt.test.ContinuousPeaksTest

This class will automatically loop through different values of N and T, and find the averages of each N and T run.

The relevant times and accuracies will be printed to stdout.

## MaxKColor

To run tne MaxKColor Problem, run the modified MaxKColoringTest.java class via:
java -cp ABAGAIL.jar opt.test.MaxKColoringTest

ThiThe following desribes how to run the Random Optmization algorithms from the assignment.

First, ABAGAIL must be downloaded from the following:

Copy the following 4 files into the opt/test directory and rebuild ABAGAIL.tar via ANT:
1. WineTestSample.java
2. ContinuousPeaksTest.java (an alteration of the original ContinuousPeaksTest.java)
3. MaxKColoringTest.java (an alteration of the original MaxKColoringTest.java)
4. TravelingSalesmanProblem.java (an alteration of the original ContinuousPeaksTest.java)

## Neural Network
To run the Neural Network tests, run WineTestSample via:
java -cp ABAGAIL.jar opt.test.WineTestSample

Note that to run Randomized Hill Climbing, uncomment out line 63 and comment out lines 66,69
Note that to run Simulated Annealing, uncomment out line 66 and comment out lines 63,69
Note that to run Genetic Algorithm, uncomment out line 69 and comment out lines 63,66

Also note that line 48 should be altered to a string that represrents the name for printing purposes (e.g. "RHC")

## Continuous Peaks

To run Continuous Peaks Problem, run the modified ContinuousPeaksTest.java file via:
java -cp ABAGAIL.jar opt.test.ContinuousPeaksTest

This class will automatically loop through different values of N and T, and find the averages of each N and T run.

The relevant times and accuracies will be printed to stdout.

## MaxKColor

To run tne MaxKColor Problem, run the modified MaxKColoringTest.java class via:
java -cp ABAGAIL.jar opt.test.MaxKColoringTest

This class will automatically iterate 10 times.  Alter K, N,  and L, alter lines: 42-45

The relevant times and accuracies will be printed to stdout.

## Traveling Salesman Problem

To run Traveling Salesman Problem, run the modified TravelingSalesmanProblem file via:
java -cp ABAGAIL.jar opt.test.TravelingSalesmanProblem.java

This class will automatically iterate through various values of N.

The relevant times and accuracies will be printed to stdout.
