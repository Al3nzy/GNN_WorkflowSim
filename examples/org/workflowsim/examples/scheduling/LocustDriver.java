package org.workflowsim.examples.scheduling;

public class LocustDriver {
	  public static void main(String[] args) {
	        // Example: 30 population, 50 iterations, 100 tasks, 10 VMs
	        LocustScheduler liwsa = new LocustScheduler(
	            30, 50,     // population size, max iterations
	            100, 10,    // numTasks, numVMs (should match DAX)
	            1.0, 0.5, 0.3, // F, L, alpha
	            0.5, 0.2,      // lambda, mutationRate
	            0.25, 0.1      // dThreshold, pCrossover
	        );

	        liwsa.run();

	        Schedule best = liwsa.getBestSchedule();
	        System.out.println("==== FINAL BEST SCHEDULE ====");
	        System.out.println("Fitness: " + best.getFitness());
	        System.out.println("Makespan: " + best.getMakespan());
	        System.out.println("Cost: " + best.getCost());
	    }
	
}
