package opt.test;

import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class ContinuousPeaksTest {
    /** The n value */
    private static int N = 60;
    /** The t value */
    private static int T = N / 10;

    public static void main(String[] args) {
        for (int h = 0; h < 5; h++) {
            N = 40 + h * 20;
            for (int i = 0; i < 5; i++) {
                T = i + 2;
                int[] ranges = new int[N];
                Arrays.fill(ranges, 2);
                EvaluationFunction ef = new ContinuousPeaksEvaluationFunction(T);
                Distribution odd = new DiscreteUniformDistribution(ranges);
                NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
                MutationFunction mf = new DiscreteChangeOneMutation(ranges);
                CrossoverFunction cf = new SingleCrossOver();
                Distribution df = new DiscreteDependencyTree(.1, ranges);
                HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
                GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
                ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

                int ites = 1;
                double optimaTotal = 0;
                float totalTime = 0;
                for (int j = 0; j < ites; j++) {
                    RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
                    FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
                    long starttime = System.currentTimeMillis();
                    fit.train();
                    optimaTotal += ef.value(rhc.getOptimal());
                    totalTime += (System.currentTimeMillis() - starttime);
                }
                System.out.println(N + "," + T  + "," + "RHC," + optimaTotal / ites + "," + totalTime / ites);

                optimaTotal = 0;
                totalTime = 0;

                for (int j = 0; j < ites; j++) {
                    SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
                    FixedIterationTrainer fit = new FixedIterationTrainer(sa, 200000);
                    long starttime = System.currentTimeMillis();
                    fit.train();
                    optimaTotal += ef.value(sa.getOptimal());
                    totalTime += (System.currentTimeMillis() - starttime);
                }
                System.out.println(N + "," + T  + "," + "SA," + optimaTotal / ites + "," + totalTime / ites);

                optimaTotal = 0;
                totalTime = 0;

                for (int j = 0; j < ites; j++) {
                    StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 10, gap);
                    FixedIterationTrainer fit = new FixedIterationTrainer(ga, 1000);
                    long starttime = System.currentTimeMillis();
                    fit.train();
                    optimaTotal += ef.value(ga.getOptimal());
                    totalTime += (System.currentTimeMillis() - starttime);
                }
                System.out.println(N + "," + T  + "," + "GA," + optimaTotal / ites + "," + totalTime / ites);

                optimaTotal = 0;
                totalTime = 0;

                for (int j = 0; j < ites; j++) {
                    MIMIC mimic = new MIMIC(200, 20, pop);
                    FixedIterationTrainer fit = new FixedIterationTrainer(mimic, 1000);
                    long starttime = System.currentTimeMillis();
                    fit.train();
                    optimaTotal += ef.value(mimic.getOptimal());
                    totalTime += (System.currentTimeMillis() - starttime);
                }
                System.out.println(N + "," + T  + "," + "MIMIC," + optimaTotal / ites + "," + totalTime / ites);
            }
        }
    }

}
