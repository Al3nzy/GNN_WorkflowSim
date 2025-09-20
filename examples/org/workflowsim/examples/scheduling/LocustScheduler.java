package org.workflowsim.examples.scheduling;

import java.util.*;
import java.util.Arrays;

public class LocustScheduler {

    private List<Schedule> population;
    private Schedule globalBest;
    private Random rng;   // single random instance

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

        this.rng = new Random(42); // fixed seed for reproducibility
    }

    public void run() {
        initializePopulation();

        for (int iter = 0; iter < maxIterations; iter++) {
            double avgDist = Utils.averagePairwiseDistance(population);

            if (avgDist > dThreshold) { // Social phase
                for (int i = 0; i < populationSize; i++) {
                    Schedule Xi = population.get(i);
                    Schedule Y = Utils.tournamentSelection(population, 3, rng);
                    double influence = computeInfluence(Xi, Y);
                    Schedule Xnew = copyByProbability(Xi, Y, influence);

                    if (rng.nextDouble() < pCrossover) {
                        Xnew = crossover(Xnew, Y);
                    }

                    repair(Xnew);
                    evaluateAndUpdate(i, Xi, Xnew);
                }
            } else { // Solitary phase
                for (int i = 0; i < populationSize; i++) {
                    Schedule Xi = population.get(i);
                    Schedule Xnew = mutate(Xi);
                    repair(Xnew);
                    evaluateAndUpdate(i, Xi, Xnew);
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
            Schedule s = new Schedule(numTasks, numVMs, rng);
            s.randomInitialize();

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
            if (rng.nextDouble() < pk) {
                copy.getAssignment()[k] = Y.getAssignment()[k];
            }
        }
        return copy;
    }

    private Schedule mutate(Schedule Xi) {
        Schedule copy = Xi.deepCopy();
        if (rng.nextDouble() < mutationRate) {
            int task = Utils.randInt(0, numTasks - 1, rng);
            int newVm = Utils.randomVM(numVMs, copy.getAssignment()[task], rng);
            copy.getAssignment()[task] = newVm;
        }
        return copy;
    }

    private Schedule crossover(Schedule X1, Schedule X2) {
        Schedule child = X1.deepCopy();
        int point = Utils.randInt(0, numTasks - 1, rng);
        for (int i = point; i < numTasks; i++) {
            child.getAssignment()[i] = X2.getAssignment()[i];
        }
        return child;
    }

    /** Ensure valid VM IDs */
    private void repair(Schedule s) {
        int[] assign = s.getAssignment();
        for (int i = 0; i < assign.length; i++) {
            if (assign[i] < 0 || assign[i] >= numVMs) {
                assign[i] = rng.nextInt(numVMs);
            }
        }
    }

    private void evaluateAndUpdate(int i, Schedule Xi, Schedule Xnew) {
        WorkflowSimEvaluator.Result r = WorkflowSimEvaluator.evaluateAssignment(Xnew.getAssignment());
        Xnew.setMakespan(r.makespan);
        Xnew.setCost(r.cost);
        Xnew.setFitness(r.fitness);

        if (Xnew.getFitness() < Xi.getFitness()) {
            population.set(i, Xnew);
        }
    }

    public Schedule getBestSchedule() {
        return globalBest;
    }
}

// ================= Schedule class =================
class Schedule {
    private int[] assignment;  
    private double makespan, cost, fitness;
    private int numVMs;
    private Random rng;

    public Schedule(int numTasks, int numVMs, Random rng) {
        this.assignment = new int[numTasks];
        this.numVMs = numVMs;
        this.rng = rng;
    }

    public void randomInitialize() {
        for (int i = 0; i < assignment.length; i++) {
            assignment[i] = rng.nextInt(numVMs);
        }
    }

    // ===== Getters & Setters =====
    public int[] getAssignment() { return assignment; }
    public double getMakespan() { return makespan; }
    public double getCost() { return cost; }
    public double getFitness() { return fitness; }
    public void setMakespan(double makespan) { this.makespan = makespan; }
    public void setCost(double cost) { this.cost = cost; }
    public void setFitness(double fitness) { this.fitness = fitness; }

    public Schedule deepCopy() {
        Schedule copy = new Schedule(assignment.length, numVMs, rng);
        copy.assignment = Arrays.copyOf(this.assignment, this.assignment.length);
        copy.makespan = this.makespan;
        copy.cost = this.cost;
        copy.fitness = this.fitness;
        return copy;
    }
}

// ================= Utils class =================
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

    public static Schedule tournamentSelection(List<Schedule> population, int k, Random rng) {
        List<Schedule> candidates = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            candidates.add(population.get(rng.nextInt(population.size())));
        }
        return Collections.min(candidates, Comparator.comparingDouble(Schedule::getFitness));
    }

    public static double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    public static int randomVM(int numVMs, int currentVm, Random rng) {
        int vm;
        do {
            vm = rng.nextInt(numVMs);
        } while (vm == currentVm);
        return vm;
    }

    public static int randInt(int min, int max, Random rng) {
        return rng.nextInt((max - min) + 1) + min;
    }
}
