package org.workflowsim.examples.scheduling;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Main Locust-Inspired Workflow Scheduling Algorithm (LIWSA).
 * Works with WorkflowSim via the WorkflowSimEvaluator class.
 */
public class LocustScheduler {

    private List<Schedule> population;
    private Schedule globalBest;

    private int populationSize, maxIterations;
    private int numTasks, numVMs;
    private double F, L, alpha, lambda, mutationRate, dThreshold, pCrossover;

    public LocustScheduler(int populationSize, int maxIterations,
                           int numTasks, int numVMs,
                           double F, double L, double alpha,
                           double lambda, double mutationRate,
                           double dThreshold, double pCrossover) {
        this.populationSize = populationSize;
        this.maxIterations = maxIterations;
        this.numTasks = numTasks;
        this.numVMs = numVMs;
        this.F = F;
        this.L = L;
        this.alpha = alpha;
        this.lambda = lambda;
        this.mutationRate = mutationRate;
        this.dThreshold = dThreshold;
        this.pCrossover = pCrossover;
    }

    public void run() {
        initializePopulation();

        for (int iter = 0; iter < maxIterations; iter++) {
            double avgDist = Utils.averagePairwiseDistance(population);

            if (avgDist > dThreshold) { // Social phase
                for (int i = 0; i < populationSize; i++) {
                    Schedule Xi = population.get(i);
                    Schedule Y = Utils.tournamentSelection(population, 3);
                    double influence = computeInfluence(Xi, Y);
                    Schedule Xnew = copyByProbability(Xi, Y, influence);

                    if (Math.random() < pCrossover) {
                        Xnew = crossover(Xnew, Y);
                    }

                    repair(Xnew);

                    // ===== Evaluate using WorkflowSimEvaluator =====
                    WorkflowSimEvaluator.Result r = WorkflowSimEvaluator.evaluateAssignment(Xnew.getAssignment());
                    Xnew.setMakespan(r.makespan);
                    Xnew.setCost(r.cost);
                    Xnew.setFitness(r.fitness);

                    if (Xnew.getFitness() < Xi.getFitness()) {
                        population.set(i, Xnew);
                    }
                }
            } else { // Solitary phase
                for (int i = 0; i < populationSize; i++) {
                    Schedule Xi = population.get(i);
                    Schedule Xnew = mutate(Xi);
                    repair(Xnew);

                    // ===== Evaluate using WorkflowSimEvaluator =====
                    WorkflowSimEvaluator.Result r = WorkflowSimEvaluator.evaluateAssignment(Xnew.getAssignment());
                    Xnew.setMakespan(r.makespan);
                    Xnew.setCost(r.cost);
                    Xnew.setFitness(r.fitness);

                    if (Xnew.getFitness() < Xi.getFitness()) {
                        population.set(i, Xnew);
                    }
                }
            }

            // Update global best
            Schedule bestInPop = Collections.min(population, Comparator.comparingDouble(Schedule::getFitness));
            if (globalBest == null || bestInPop.getFitness() < globalBest.getFitness()) {
                globalBest = bestInPop.deepCopy();
            }

            // Log progress
            System.out.printf("Iter %d: Best Fitness=%.4f, Makespan=%.2f, Cost=%.2f%n",
                    iter, globalBest.getFitness(), globalBest.getMakespan(), globalBest.getCost());
        }

        System.out.println("Final Best Schedule:");
        System.out.println("Fitness=" + globalBest.getFitness() +
                           " Makespan=" + globalBest.getMakespan() +
                           " Cost=" + globalBest.getCost());
    }

    private void initializePopulation() {
        population = new ArrayList<>();
        for (int i = 0; i < populationSize; i++) {
            Schedule s = new Schedule(numTasks, numVMs);
            s.randomInitialize();

            // ===== Evaluate using WorkflowSimEvaluator =====
            WorkflowSimEvaluator.Result r = WorkflowSimEvaluator.evaluateAssignment(s.getAssignment());
            s.setMakespan(r.makespan);
            s.setCost(r.cost);
            s.setFitness(r.fitness);

            population.add(s);
        }
        globalBest = Collections.min(population, Comparator.comparingDouble(Schedule::getFitness));
    }

    private double computeInfluence(Schedule Xi, Schedule Y) {
        double d = Utils.hammingDistance(Xi, Y);
        double z = F * Math.exp(-d / L) - Math.exp(-d);
        return Utils.sigmoid(z);
    }

    private Schedule copyByProbability(Schedule Xi, Schedule Y, double influence) {
        Schedule copy = Xi.deepCopy();
        for (int k = 0; k < numTasks; k++) {
            double pk = alpha * influence * (lambda + (1 - lambda));
            if (Math.random() < pk) {
                copy.getAssignment()[k] = Y.getAssignment()[k];
            }
        }
        return copy;
    }

    private Schedule mutate(Schedule Xi) {
        Schedule copy = Xi.deepCopy();
        if (Math.random() < mutationRate) {
            int task = Utils.randInt(0, numTasks - 1);
            int newVm = Utils.randomVM(numVMs, copy.getAssignment()[task]);
            copy.getAssignment()[task] = newVm;
        }
        return copy;
    }

    private Schedule crossover(Schedule X1, Schedule X2) {
        Schedule child = X1.deepCopy();
        int point = Utils.randInt(0, numTasks - 1);
        for (int i = point; i < numTasks; i++) {
            child.getAssignment()[i] = X2.getAssignment()[i];
        }
        return child;
    }

    private void repair(Schedule s) {
        // Currently, WorkflowSimEvaluator handles task dependencies.
    }

    public Schedule getBestSchedule() {
        return globalBest;
    }
}

/**
 * Represents a candidate schedule.
 */
class Schedule {
    private int[] assignment;  // task -> VM mapping
    private double makespan;
    private double cost;
    private double fitness;

    private int numVMs;

    public Schedule(int numTasks, int numVMs) {
        this.assignment = new int[numTasks];
        this.numVMs = numVMs;
    }

    public void randomInitialize() {
        Random rand = new Random();
        for (int i = 0; i < assignment.length; i++) {
            assignment[i] = rand.nextInt(numVMs);
        }
    }

    public int[] getAssignment() { return assignment; }
    public double getMakespan() { return makespan; }
    public double getCost() { return cost; }
    public double getFitness() { return fitness; }

    public void setMakespan(double makespan) { this.makespan = makespan; }
    public void setCost(double cost) { this.cost = cost; }
    public void setFitness(double fitness) { this.fitness = fitness; }

    public Schedule deepCopy() {
        Schedule copy = new Schedule(assignment.length, numVMs);
        copy.assignment = Arrays.copyOf(this.assignment, this.assignment.length);
        copy.makespan = this.makespan;
        copy.cost = this.cost;
        copy.fitness = this.fitness;
        return copy;
    }
}

/**
 * Utilities for distances, randomness, and selection.
 */
class Utils {
    public static double hammingDistance(Schedule X, Schedule Y) {
        int[] a = X.getAssignment();
        int[] b = Y.getAssignment();
        int diff = 0;
        for (int i = 0; i < a.length; i++) {
            if (a[i] != b[i]) diff++;
        }
        return (double) diff / a.length;
    }

    public static double averagePairwiseDistance(List<Schedule> population) {
        double sum = 0.0;
        int count = 0;
        for (int i = 0; i < population.size(); i++) {
            for (int j = i + 1; j < population.size(); j++) {
                sum += hammingDistance(population.get(i), population.get(j));
                count++;
            }
        }
        return count > 0 ? sum / count : 0.0;
    }

    public static Schedule tournamentSelection(List<Schedule> population, int k) {
        Random rand = new Random();
        List<Schedule> candidates = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            candidates.add(population.get(rand.nextInt(population.size())));
        }
        return Collections.min(candidates, Comparator.comparingDouble(Schedule::getFitness));
    }

    public static double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    public static int randomVM(int numVMs, int currentVm) {
        Random rand = new Random();
        int vm;
        do {
            vm = rand.nextInt(numVMs);
        } while (vm == currentVm);
        return vm;
    }

    public static int randInt(int min, int max) {
        return new Random().nextInt((max - min) + 1) + min;
    }
}
