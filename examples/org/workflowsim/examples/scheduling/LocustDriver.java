package org.workflowsim.examples.scheduling;

import java.io.File;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.util.List;

import org.workflowsim.Task;
import org.workflowsim.utils.ClusteringParameters;
import org.workflowsim.utils.OverheadParameters;
import org.workflowsim.utils.Parameters;
import org.workflowsim.utils.ReplicaCatalog;

/**
 * LocustDriver: initialize parameters, run Locust (with surrogate evaluations),
 * then run a final full CloudSim simulation using the best assignment.
 */
public class LocustDriver {

    public static final String DAX_PATH = "examples/org/workflowsim/examples/workflowDatasets/Montage_50.xml";

    public static void main(String[] args) {
        try {
            System.out.println("=== LocustDriver starting ===");

            // 1) Check DAX exists
            File daxFile = new File(DAX_PATH);
            if (!daxFile.exists()) {
                System.err.println("DAX file not found: " + DAX_PATH);
                return;
            }

            // 2) Initialize WorkflowSim Parameters & ReplicaCatalog
            int vmNum = 5;
            OverheadParameters op = new OverheadParameters(0, null, null, null, null, 0);
            ClusteringParameters cp = new ClusteringParameters(0, 0,
                    ClusteringParameters.ClusteringMethod.NONE, null);

            Parameters.init(vmNum, DAX_PATH, null, null,
                    op, cp, Parameters.SchedulingAlgorithm.INVALID,
                    Parameters.PlanningAlgorithm.INVALID, null, 0L);

            ReplicaCatalog.init(ReplicaCatalog.FileSystem.LOCAL);

            // 3) Parse DAX (reflectively) to count tasks
            int parsedNumTasks = parseDaxTaskCount(0);
            System.out.println("Parsed number of tasks from DAX: " + parsedNumTasks);
            if (parsedNumTasks <= 0) {
                System.err.println("No tasks parsed; aborting.");
                return;
            }

            // 4) Configure and run LocustScheduler using surrogate evaluations
            int populationSize = 30;
            int maxIterations = 5; // reduce while testing
            int numTasks = parsedNumTasks;
            int numVMs = vmNum;

            // ensure evaluator uses surrogate during optimization
            WorkflowSimEvaluator.FULL_SIM = false;
            WorkflowSimEvaluator.PRINT_CLOUDLETS = false;

            LocustScheduler locust = new LocustScheduler(
                    populationSize, maxIterations,
                    numTasks, numVMs,
                    1.0, 0.5, 0.3,    // F, L, alpha
                    0.5, 0.2,         // lambda, mutationRate
                    0.25, 0.1         // dThreshold, pCrossover
            );

            System.out.println("Running LocustScheduler (surrogate evaluations)...");
            locust.run();

            // 5) Final run: run a single full CloudSim simulation with the best schedule
            Schedule best = locust.getBestSchedule();
            if (best != null) {
                System.out.println("\n=== Final run using best schedule (full CloudSim) ===");
                WorkflowSimEvaluator.FULL_SIM = true;
                WorkflowSimEvaluator.PRINT_CLOUDLETS = true;
                WorkflowSimEvaluator.Result finalRes = WorkflowSimEvaluator.evaluateAssignment(best.getAssignment());

                System.out.println("---------- FINAL METRICS (Locust) ----------");
                System.out.printf(
                    "Fitness=%.6f%nMakespan=%.3f%nCost=%.3f%nUtilization=%.3f%nThroughput=%.3f%nAvgResponse=%.3f%nAvgWaiting=%.3f%nLoadBalance=%.3f%n",
                    finalRes.fitness, finalRes.makespan, finalRes.cost,
                    finalRes.utilization, finalRes.throughput,
                    finalRes.avgResponse, finalRes.avgWaiting, finalRes.loadBalance
                );

            } else {
                System.out.println("Locust returned no best schedule.");
            }

            System.out.println("=== LocustDriver finished ===");
        } catch (Throwable t) {
            t.printStackTrace();
            System.err.println("Driver terminated with error: " + t.getMessage());
        }
    }

    /** Helper: parse DAX and return number of tasks via WorkflowParser reflectively */
    private static int parseDaxTaskCount(int userId) throws Exception {
        Class<?> parserClass = Class.forName("org.workflowsim.WorkflowParser");
        Constructor<?> ctor = parserClass.getConstructor(int.class);
        Object parserInstance = ctor.newInstance(userId);
        Method parseMethod = parserClass.getMethod("parse");
        parseMethod.invoke(parserInstance);
        Method getTaskListMethod = parserClass.getMethod("getTaskList");
        @SuppressWarnings("unchecked")
        List<Task> taskList = (List<Task>) getTaskListMethod.invoke(parserInstance);
        return taskList == null ? 0 : taskList.size();
    }
}
