package org.workflowsim.examples.scheduling;

import java.util.*;

/**
 * LocustScheduler: discrete locust-inspired search in task->VM assignment space.
 * Calls WorkflowSimEvaluator (surrogate mode during search).
 */
public class LocustScheduler {

    private List<Schedule> population;
    private Schedule globalBest;
    private Random rng;

    private int populationSize, maxIter;
    private int numTasks, numVMs;
    private double F, L, alpha, lambda, mutationRate, dThreshold, pCrossover;

    public LocustScheduler(int populationSize, int maxIter,
                           int numTasks, int numVMs,
                           double F, double L, double alpha,
                           double lambda, double mutationRate,
                           double dThreshold, double pCrossover) {
        this.populationSize = populationSize;
        this.maxIter = maxIter;
        this.numTasks = numTasks;
        this.numVMs = numVMs;
        this.F = F;
        this.L = L;
        this.alpha = alpha;
        this.lambda = lambda;
        this.mutationRate = mutationRate;
        this.dThreshold = dThreshold;
        this.pCrossover = pCrossover;
        this.rng = new Random(42);
    }

    public void run() {
        initializePopulation();

        for (int iter = 0; iter < maxIter; iter++) {
            double avgDist = Utils.averagePairwiseDistance(population);

            if (avgDist > dThreshold) { // social
                for (int i = 0; i < populationSize; i++) {
                    Schedule Xi = population.get(i);
                    Schedule Y = Utils.tournamentSelection(population, 3, rng);
                    double influence = computeInfluence(Xi, Y);
                    Schedule Xnew = copyByProbability(Xi, Y, influence);

                    if (rng.nextDouble() < pCrossover) Xnew = crossover(Xnew, Y);

                    repair(Xnew);

                    evaluateAndReplace(i, Xi, Xnew);
                }
            } else { // solitary
                for (int i = 0; i < populationSize; i++) {
                    Schedule Xi = population.get(i);
                    Schedule Xnew = mutate(Xi);
                    repair(Xnew);
                    evaluateAndReplace(i, Xi, Xnew);
                }
            }

            // update global best
            Schedule bestInPop = Collections.min(population, Comparator.comparingDouble(Schedule::getFitness));
            if (globalBest == null || bestInPop.getFitness() < globalBest.getFitness()) {
                globalBest = bestInPop.deepCopy();
            }

            System.out.printf("Iter %d: Best Fitness=%.4f, Makespan=%.2f, Cost=%.2f%n",
                    iter, globalBest.getFitness(), globalBest.getMakespan(), globalBest.getCost());
        }

        System.out.println("Final Best Schedule:");
        System.out.println("Fitness=" + globalBest.getFitness() + " Makespan=" + globalBest.getMakespan() + " Cost=" + globalBest.getCost());
    }

    private void initializePopulation() {
        population = new ArrayList<>();
        for (int i = 0; i < populationSize; i++) {
            Schedule s = new Schedule(numTasks, numVMs, rng);
            s.randomInitialize();
            WorkflowSimEvaluator.Result r = WorkflowSimEvaluator.evaluateAssignment(s.getAssignment(), numVMs);
            s.setMakespan(r.makespan);
            s.setCost(r.cost);
            s.setFitness(r.fitness);
            population.add(s);
        }
        globalBest = Collections.min(population, Comparator.comparingDouble(Schedule::getFitness));
    }

    private double computeInfluence(Schedule a, Schedule b) {
        double d = Utils.hammingDistance(a, b);
        double z = F * Math.exp(-d / L) - Math.exp(-d);
        return Utils.sigmoid(z);
    }

    private Schedule copyByProbability(Schedule Xi, Schedule Y, double influence) {
        Schedule copy = Xi.deepCopy();
        for (int k = 0; k < numTasks; k++) {
            double pk = alpha * influence * (lambda + (1 - lambda) * 1.0); // no per-task weight now
            if (rng.nextDouble() < pk) copy.getAssignment()[k] = Y.getAssignment()[k];
        }
        return copy;
    }

    private Schedule mutate(Schedule Xi) {
        Schedule copy = Xi.deepCopy();
        if (rng.nextDouble() < mutationRate) {
            int t = Utils.randInt(0, numTasks - 1, rng);
            int newVm = Utils.randomVM(numVMs, copy.getAssignment()[t], rng);
            copy.getAssignment()[t] = newVm;
        }
        return copy;
    }

    private Schedule crossover(Schedule a, Schedule b) {
        Schedule child = a.deepCopy();
        int point = Utils.randInt(0, numTasks - 1, rng);
        for (int i = point; i < numTasks; i++) child.getAssignment()[i] = b.getAssignment()[i];
        return child;
    }

    private void repair(Schedule s) {
        int[] asn = s.getAssignment();
        for (int i = 0; i < asn.length; i++) {
            if (asn[i] < 0 || asn[i] >= numVMs) asn[i] = rng.nextInt(numVMs);
        }
    }

    private void evaluateAndReplace(int idx, Schedule oldS, Schedule newS) {
        WorkflowSimEvaluator.Result r = WorkflowSimEvaluator.evaluateAssignment(newS.getAssignment(), numVMs);
        newS.setMakespan(r.makespan);
        newS.setCost(r.cost);
        newS.setFitness(r.fitness);
        if (newS.getFitness() < oldS.getFitness()) population.set(idx, newS);
    }

    public Schedule getBestSchedule() { return globalBest; }
}

/* Schedule and Utils classes (simple): */

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
        for (int i = 0; i < assignment.length; i++) assignment[i] = rng.nextInt(numVMs);
    }

    public int[] getAssignment() { return assignment; }
    public double getMakespan() { return makespan; }
    public double getCost() { return cost; }
    public double getFitness() { return fitness; }

    public void setMakespan(double m) { this.makespan = m; }
    public void setCost(double c) { this.cost = c; }
    public void setFitness(double f) { this.fitness = f; }

    public Schedule deepCopy() {
        Schedule copy = new Schedule(assignment.length, numVMs, rng);
        copy.assignment = Arrays.copyOf(this.assignment, this.assignment.length);
        copy.makespan = this.makespan;
        copy.cost = this.cost;
        copy.fitness = this.fitness;
        return copy;
    }
}

class Utils {
    public static double hammingDistance(Schedule a, Schedule b) {
        int[] x = a.getAssignment(), y = b.getAssignment();
        int diff = 0;
        for (int i = 0; i < x.length; i++) if (x[i] != y[i]) diff++;
        return (double) diff / x.length;
    }

    public static double averagePairwiseDistance(List<Schedule> pop) {
        double sum = 0; int count = 0;
        for (int i = 0; i < pop.size(); i++) for (int j = i+1; j < pop.size(); j++) { sum += hammingDistance(pop.get(i), pop.get(j)); count++; }
        return count>0 ? sum/count : 0;
    }

    public static Schedule tournamentSelection(List<Schedule> pop, int k, Random rng) {
        Schedule best = null;
        for (int i = 0; i < k; i++) {
            Schedule c = pop.get(rng.nextInt(pop.size()));
            if (best == null || c.getFitness() < best.getFitness()) best = c;
        }
        return best;
    }

    public static double sigmoid(double z) { return 1.0 / (1.0 + Math.exp(-z)); }

    public static int randomVM(int numVMs, int current, Random rng) {
        int v;
        do { v = rng.nextInt(numVMs); } while (v == current);
        return v;
    }

    public static int randInt(int min, int max, Random rng) { return rng.nextInt((max-min)+1) + min; }
}
